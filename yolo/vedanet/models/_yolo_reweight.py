import os
from collections import OrderedDict, Iterable
import torch
import time
import copy
import torch.nn as nn
from .. import loss
import pickle
from .yolo_abc import YoloABC
from ..network import backbone
from ..network import head
from ..network import metanet

__all__ = ['Yolov2_Meta', 'TinyYolov2_Meta']


class Yolov2_Meta(YoloABC):
    def __init__(self, num_classes=20, weights_file=None, input_channels=3,
                 anchors=[(42.31, 55.41), (102.17, 128.30), (161.79, 259.17), (303.08, 154.90), (359.56, 320.23)],
                 anchors_mask=[(0, 1, 2, 3, 4)], train_flag=1, clear=False, test_args=None, reweights_file=None,
                 tiny_backbone=False):
        """ Network initialisation """
        super().__init__()

        # Parameters
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.nloss = len(self.anchors_mask)
        self.train_flag = train_flag
        self.test_args = test_args

        self.last_layer = None

        self.loss = None
        self.postprocess = None
        self.loss_fn = loss.RepLoss
        if tiny_backbone:
            self.backbone = backbone.NanoYolov2()
        else:
            self.backbone = backbone.Darknet19()
        if torch.cuda.device_count() > 1:
            self.dist_backbone = torch.nn.DataParallel(self.backbone)
        else:
            self.dist_backbone = self.backbone
        self.head = head.MetaYolov2(num_anchors=len(anchors_mask[0]), num_classes=num_classes)
        # self.metanet = metanet.Metanet(num_classes=num_classes)
        if train_flag == 2:
            if reweights_file is not None:
                with open(reweights_file, 'rb') as handle:
                    reweights = pickle.load(handle)
                    print(reweights_file)
                    self.reweights = torch.Tensor(len(reweights.keys()), *reweights[0].shape).cuda()
                    for i in range(len(reweights.keys())):
                        self.reweights[i] = reweights[i]
            else:
                exit(0)

        if weights_file is not None:
            self.load_weights(weights_file, clear)
        else:
            self.init_weights(slope=0.1)

    def _forward(self, x):
        data, reweights = x
        middle_feats = self.dist_backbone(data)
        # reweights = self.metanet(meta_imgs)
        features = self.head(middle_feats, reweights)
        self.compose(data, features, self.loss_fn)
        return features

    def _forward_test(self, x, reweights):

        data = x
        middle_feats = self.backbone(data)
        features = self.head(middle_feats, reweights)

        self.compose(data, features, self.loss_fn)

        return [self.convert_to_yolo_output(f) for f in features]

    def forward(self, x, target=None):
        if self.train_flag == 1:
            x, reweights = x
            self.seen += x.size(0)
            t1 = time.time()
            outputs = self._forward((x, reweights))
            t2 = time.time()

            assert len(outputs) == len(self.loss)

            loss = 0
            for idx in range(len(outputs)):
                assert callable(self.loss[idx])
                t1 = time.time()
                loss += self.loss[idx](outputs[idx], target)
                t2 = time.time()
            return loss
        else:
            t1 = time.time()
            outputs = self._forward_test(x, reweights=self.reweights)
            if self.postprocess is None:
                return  # speed
            t2 = time.time()
            print('forward took {:.5f}s'.format(t2 - t1))
            loss = None
            dets = []

            tdets = []
            for idx in range(len(outputs)):
                assert callable(self.postprocess[idx])
                tdets.append(self.postprocess[idx](outputs[idx]))

            batch = len(tdets[0])
            for b in range(batch):
                single_dets = []
                for op in range(len(outputs)):
                    single_dets.extend(tdets[op][b])
                dets.append(single_dets)
            t3 = time.time()
            print('postprocessing took {:.5f}s'.format(t3 - t2))
            if loss is not None:
                return dets, loss
            else:
                return dets, 0.0


    def modules_recurse(self, mod=None):
        """ This function will recursively loop over all module children.

        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**
        """
        if mod is None:
            mod = self

        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential, backbone.Darknet19, head.Yolov2)):
                yield from self.modules_recurse(module)
            else:
                yield module

    def convert_to_yolo_output(self, prediction):
        if self.num_classes == 1:
            return prediction
        batch_size = int(prediction.shape[0] / self.num_classes)
        grouped_prediction = prediction.view(batch_size, self.num_classes, 5, 6, *prediction.shape[2:])
        grouped_prediction = grouped_prediction.permute(0, 2, 1, 3, 4, 5)               # (batch, num_anchors, num_classes, 6, h, w)

        class_score = grouped_prediction[:, :, :, 5:6, :, :]
        max_class_probs, max_ids = torch.max(class_score, dim=2, keepdim=True)

        gather_ids = max_ids.repeat(1, 1, 1, 5, 1, 1)
        final_prediction_0_5 = torch.gather(grouped_prediction[:, :, :, 0:5, :, :],
                                        dim=2, index=gather_ids).squeeze(dim=2)
        final_prediction_5 = class_score.squeeze(dim=3)
        final_prediction = torch.cat([final_prediction_0_5,
                                      final_prediction_5],
                                     dim=2)
        reshaped_final_prediction = final_prediction.view(batch_size, -1, *final_prediction.shape[-2:] )
        assert not torch.isnan(reshaped_final_prediction).any()
        return reshaped_final_prediction


class TinyYolov2_Meta(Yolov2_Meta):
    def __init__(self, num_classes=20, weights_file=None, input_channels=3,
                 anchors=[(42.31, 55.41), (102.17, 128.30), (161.79, 259.17), (303.08, 154.90), (359.56, 320.23)],
                 anchors_mask=[(0, 1, 2, 3, 4)], train_flag=1, clear=False, test_args=None, reweights_file=None):
        super(TinyYolov2_Meta, self).__init__(num_classes, weights_file, input_channels,
                                              anchors,
                                              anchors_mask, train_flag, clear, test_args, reweights_file,
                                              tiny_backbone=True)
