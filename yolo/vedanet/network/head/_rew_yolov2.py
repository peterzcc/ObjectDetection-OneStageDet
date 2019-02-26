import os
from collections import OrderedDict
import torch
import torch.nn as nn

from .. import layer as vn_layer

__all__ = ['RewYolov2']





class RewYolov2(nn.Module):
    def __init__(self, num_anchors, num_classes, input_channels=48):
        """ Network initialisation """
        super().__init__()
        layer_list = [
            # Sequence 2 : input = sequence0
            OrderedDict([
                ('1_convbatch',    vn_layer.Conv2dBatchLeaky(512, 64, 1, 1)),
                ('2_reorg',        vn_layer.Reorg(2)),
                ]),

            # Sequence 3 : input = sequence2 + sequence1
            OrderedDict([
                ('3_convbatch',    vn_layer.Conv2dBatchLeaky((4*64)+1024, 1024, 3, 1)),
                ]),

            OrderedDict([
                ('4_conv', nn.Conv2d(1024, num_anchors * (5 + 1), 1, 1, 0)),
            ]),
            ]
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])
        self.reweighting_layer = None

    def get_test_reweighting_layer(self, device):
        reweighting_layer = torch.arange(0, self.num_classes, dtype=torch.float32, device=device)
        # do modulo:
        reweighting_layer = reweighting_layer.fmod(self.num_classes)
        reweighting_layer = reweighting_layer.view(1, self.num_classes, 1, 1, 1)
        return reweighting_layer

    def get_reweighted_output(self, pre_ultimate_layer: torch.Tensor, detector, reweighting_layer):
        batch_size = pre_ultimate_layer.shape[0]
        # (16,1024,19,19)
        tiled_preout = pre_ultimate_layer.unsqueeze(1).repeat(1, self.num_classes,1,1,1)

        reweighted_preout = tiled_preout * reweighting_layer
        reshaped_preout = reweighted_preout.view(-1, *reweighted_preout.shape[2:])
        prediction = detector(reshaped_preout)
        grouped_prediction = prediction.view(batch_size, -1, self.num_classes, 6, *prediction.shape[2:])

        f_softmax_axis = torch.nn.Softmax(dim=2)
        softmaxed_grouped_prediction = torch.cat([grouped_prediction[:, :, :, 0:5, :, :],
                                                  f_softmax_axis(grouped_prediction[:, :, :, 5:6, :, :])],
                                                 dim=3)
        agg_shape = [batch_size,grouped_prediction.shape[1],
                     5 + self.num_classes,*grouped_prediction.shape[4:]]
        class_probs = softmaxed_grouped_prediction[:, :, :, 5, :, :]
        max_class_probs, max_ids = torch.max(class_probs, dim=2, keepdim=True)
        max_class_probs = max_class_probs.repeat(1, 1, self.num_classes, 1, 1)
        is_valid = (class_probs== max_class_probs).unsqueeze(3).repeat(1, 1, 1, 5, 1, 1)
        final_prediction_0_5 = (softmaxed_grouped_prediction[:, :, :, 0:5, :, :]*is_valid.float()).sum(dim=2)
        final_prediction_5 = class_probs
        final_prediction = torch.cat([final_prediction_0_5,
                                      final_prediction_5],
                                     dim=2)
        # reshaped_final_prediction = final_prediction.transpose(1, 2).contiguous() #should I transpose?
        reshaped_final_prediction = final_prediction
        reshaped_final_prediction = reshaped_final_prediction.view(batch_size, -1, *final_prediction.shape[-2:] )
        assert not torch.isnan(reshaped_final_prediction).any()
        return reshaped_final_prediction

    def forward(self, middle_feats):
        outputs = []
        # stage 5
        # Route : layers=-9
        stage6_reorg = self.layers[0](middle_feats[1])
        # stage 6
        stage6 = middle_feats[0]
        # Route : layers=-1, -4
        pre_out = self.layers[1](torch.cat((stage6_reorg, stage6), 1))
        if self.reweighting_layer is None:
            self.reweighting_layer = self.get_test_reweighting_layer(device=pre_out.device)
        out = self.get_reweighted_output(pre_out, self.layers[2], self.reweighting_layer)
        features = [out]
        return features
