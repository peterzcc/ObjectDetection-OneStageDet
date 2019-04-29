import os
from collections import OrderedDict
import torch
import torch.nn as nn

from .. import layer as vn_layer

__all__ = ['UniYolov2']


class UniYolov2(nn.Module):
    def __init__(self, num_anchors, num_classes, input_channels=48):
        """ Network initialisation """
        super().__init__()
        self.num_anchors = num_anchors
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
            ]
        self.pred_input_size = 512
        preout_layer = \
            OrderedDict([
                (f'4_convbatch',    vn_layer.Conv2dBatchLeaky(1024, num_anchors*self.pred_input_size, 1, 1)),
            ])
        layer_list.append(preout_layer)
        uni_predictor = \
            OrderedDict([
                ('5_conv',         nn.Conv2d(self.pred_input_size, (5+num_classes), 1, 1, 0)),
                ])
        layer_list.append(uni_predictor)
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, middle_feats):
        outputs = []
        # stage 5
        # Route : layers=-9
        stage6_reorg = self.layers[0](middle_feats[1])
        # stage 6
        stage6 = middle_feats[0]
        # Route : layers=-1, -4
        feature_layer = self.layers[1](torch.cat((stage6_reorg, stage6), 1))
        anchor_aggregated = self.layers[2](feature_layer)
        b, _, h, w = anchor_aggregated.shape
        anchor_separated = anchor_aggregated.view(b, self.num_anchors, self.pred_input_size, h, w)
        anchor_in_batch = anchor_separated.view(b*self.num_anchors, self.pred_input_size,h,w).contiguous()
        out_anchor_in_batch = self.layers[3](anchor_in_batch)
        out = out_anchor_in_batch.view(b, -1, h, w).contiguous()
        features = [out]
        return features
