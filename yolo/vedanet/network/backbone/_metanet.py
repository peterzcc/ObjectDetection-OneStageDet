#
#   Darknet Darknet19 model
#   Copyright EAVISE
#


# modified by mileistone

import os
from collections import OrderedDict
import torch
import torch.nn as nn

from .. import layer as vn_layer

__all__ = ['MetaNet']


class MetaNet(nn.Module):
    """
    """
    def __init__(self,input_channels=3):
        """ Network initialisation """
        super().__init__()

        # Network
        layer_list = [
            OrderedDict([
                ('1_convbatch',     vn_layer.Conv2dBatchLeaky(input_channels, 32, 3, 1)),
                ('2_max',           nn.MaxPool2d(2, 2)),
                ('3_convbatch',     vn_layer.Conv2dBatchLeaky(32, 64, 3, 1)),
                ('4_max',           nn.MaxPool2d(2, 2)),
                ('5_convbatch',     vn_layer.Conv2dBatchLeaky(64, 128, 3, 1)),
                ]),

            OrderedDict([
                ('6_max',           nn.MaxPool2d(2, 2)),
                ('7_convbatch',     vn_layer.Conv2dBatchLeaky(128, 256, 3, 1)),
                ]),

            OrderedDict([
                ('8_max',          nn.MaxPool2d(2, 2)),
                ('9_convbatch',    vn_layer.Conv2dBatchLeaky(256, 512, 3, 1)),
                ]),

            OrderedDict([
                ('10_max',          nn.MaxPool2d(2, 2)),
                ('11_convbatch',    vn_layer.Conv2dBatchLeaky(512, 1024, 3, 1)),
            ]),
            OrderedDict([
                ('12_max', nn.MaxPool2d(2, 2)),
                ('13_convbatch', vn_layer.Conv2dBatchLeaky(1024, 1024, 3, 1)),
            ]),
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, x):
        stem = self.layers[0](x)
        stage4 = self.layers[1](stem)
        stage5 = self.layers[2](stage4)
        stage6 = self.layers[3](stage5)
        stage7 = self.layers[4](stage6)
        features = [stage7, stage6, stage5, stage4]
        return features
