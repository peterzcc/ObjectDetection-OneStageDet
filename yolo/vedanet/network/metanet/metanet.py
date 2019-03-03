#
#   MetaNet model
#



import os
from collections import OrderedDict
import torch
import torch.nn as nn

from .. import layer as vn_layer

__all__ = ['Metanet']


class Metanet(nn.Module):
    """ `Metanet`_ implementation with pytorch.

    Args:
        weights_file (str, optional): Path to the saved weights; Default **None**
        input_channels (Number, optional): Number of input channels; Default **3**

    Attributes:
        self.loss (fn): loss function. Usually this is :class:`~lightnet.network.RegionLoss`
        self.postprocess (fn): Postprocessing function. By default this is :class:`~lightnet.data.GetBoundingBoxes`

    """
    def __init__(self, num_classes=20):
        """ Network initialisation """
        super().__init__()
        self.num_classes = num_classes
        # Network
        layer_list = [
            OrderedDict([
                ('1_convbatch',     vn_layer.Conv2dBatchLeaky(4, 32, 3, 1)),
                ('2_max',           nn.MaxPool2d(2, 2)),
                ('3_convbatch',     vn_layer.Conv2dBatchLeaky(32, 64, 3, 1)),
                ('4_max',           nn.MaxPool2d(2, 2)),
                ('5_convbatch',     vn_layer.Conv2dBatchLeaky(64, 128, 3, 1)),
                ('6_max',           nn.MaxPool2d(2, 2)),
                ('7_convbatch',     vn_layer.Conv2dBatchLeaky(128, 256, 3, 1)),
                ('8_max',           nn.MaxPool2d(2, 2)),
                ('9_convbatch',     vn_layer.Conv2dBatchLeaky(256, 512, 3, 1)),
                ('10_max',          nn.MaxPool2d(2, 2)),
                ('11_convbatch',    vn_layer.Conv2dBatchLeaky(512, 1024, 3, 1)),
                ('12_max',          nn.MaxPool2d(2, 2)),
                ('13_convbatch',    vn_layer.Conv2dBatchLeaky(1024, 1024, 3, 1)),
                ]),

            OrderedDict([
                ('14_globavgpool',  nn.AdaptiveAvgPool2d(1)),       # avg pooling to size (1 x 1)
            ])

        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, x):
        temp = [x[i] for i in range(x.shape[0])]
        x = torch.cat(temp, 0)
        print(x.shape)
        feature = self.layers[0](x)
        weights = self.layers[1](feature)
        return weights