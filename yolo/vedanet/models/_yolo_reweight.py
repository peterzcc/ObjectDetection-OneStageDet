import os
from collections import OrderedDict, Iterable
import torch
import time
import torch.nn as nn
from .. import loss
import pickle
from .yolo_abc import YoloABC
from ..network import backbone
from ..network import head
from ..network import metanet

__all__ = ['Yolov2_Meta']


class Yolov2_Meta(YoloABC):
    def __init__(self, num_classes=20, weights_file=None, input_channels=3,
                 anchors=[(42.31, 55.41), (102.17, 128.30), (161.79, 259.17), (303.08, 154.90), (359.56, 320.23)],
                 anchors_mask=[(0, 1, 2, 3, 4)], train_flag=1, clear=False, test_args=None, reweights_file=None):
        """ Network initialisation """
        super().__init__()

        # Parameters
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.nloss = len(self.anchors_mask)
        self.train_flag = train_flag
        self.test_args = test_args

        self.loss = None
        self.postprocess = None

        self.backbone = backbone.Darknet19()
        self.head = head.MetaYolov2(num_anchors=len(anchors_mask[0]), num_classes=num_classes)
        if train_flag == 1:
            self.metanet = metanet.Metanet(num_classes=num_classes)
        elif train_flag == 2:
            if reweights_file is not None:
                with open(reweights_file, 'rb') as handle:
                    reweights = pickle.load(handle)
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
        data, meta_imgs = x
        middle_feats = self.backbone(data)
        reweights = self.metanet(meta_imgs)
        features = self.head(middle_feats, reweights)
        loss_fn = loss.RegionLoss

        self.compose(data, features, loss_fn)

        return features

    def _forward_test(self, x, reweights):

        data = x
        middle_feats = self.backbone(data)
        features = self.head(middle_feats, reweights)

        loss_fn = loss.RegionLoss

        self.compose(data, features, loss_fn)

        return features

    def forward(self, x, target=None):
        if self.train_flag == 1:
            x, meta_imgs = x
            self.seen += x.size(0)
            t1 = time.time()
            outputs = self._forward((x, meta_imgs))
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
