import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm

from .. import layer as vn_layer

__all__ = ['Yolov2dist']


def get_dist_cls(input, conv_layer):
    return conv_layer(input)


class DistConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1,):
        super(DistConv, self).__init__()
        self.L = nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding, dilation, bias =False)
        WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm
        if out_channels <=200:
            self.scale_factor = 2 #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10 #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True) #.unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-15)
        L_norm = torch.norm(self.L.weight.data, p=2, dim=1, keepdim=True) #.unsqueeze(1).expand_as(self.L.weight.data)
        self.L.weight.data = self.L.weight.data.div(L_norm + 1e-15)
        cos_dist = self.L(x_normalized) #matrix product by forward function
        scores = self.scale_factor* (cos_dist)

        return scores


class Yolov2dist(nn.Module):
    def __init__(self, num_anchors, num_classes, input_channels=48):
        """ Network initialisation """
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
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
                ('4_conv', nn.Conv2d(1024, num_anchors * 5, 1, 1, 0)),
            ]),
            OrderedDict([
                ('5_conv', DistConv(1024, num_anchors * num_classes, 1, 1, 0)),
            ]),
            ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, middle_feats):
        outputs = []
        # stage 5
        # Route : layers=-9
        stage6_reorg = self.layers[0](middle_feats[1])
        # stage 6
        stage6 = middle_feats[0]
        # Route : layers=-1, -4
        preout = self.layers[1](torch.cat((stage6_reorg, stage6), 1))
        b, f, h, w = preout.shape
        out_oxywh = self.layers[2](preout).view(b, self.num_anchors, 5, h, w)
        out_cls = get_dist_cls(preout,self.layers[3]).view(b, self.num_anchors, self.num_classes, h, w)
        out = torch.cat([out_oxywh, out_cls], 2).view(b, -1, h, w) #handle cat
        features = [out]
        return features
