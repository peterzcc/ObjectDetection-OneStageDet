#
#   MetaNet model
#



import os
from collections import OrderedDict
import torch
import torch.nn as nn
import logging as log

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
    def __init__(self, num_classes=20, weights_file=None):
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
                ('14_globalmaxpool',  nn.AdaptiveMaxPool2d(1)),       # max pooling to size (1 x 1)
            ])

        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

        if weights_file is not None:
            self.load_weights(weights_file)


    def forward(self, x):
        temp = [x[i] for i in range(x.shape[0])]
        x = torch.cat(temp, 0)
        feature = self.layers[0](x)
        weights = self.layers[1](feature) + 1.0 # to test
        return weights


    def load_weights(self, weights_file):
        state = torch.load(weights_file, lambda storage, loc: storage)
        old_state = self.state_dict()

        for key in list(state['weights'].keys()):
            if 'metanet' in key:
                new_key = key.replace('metanet.', '')
                state['weights'][new_key] = state['weights'].pop(key)
            else:
                state['weights'].pop(key)

        new_state = state['weights']
        if new_state.keys() != old_state.keys():
            log.warn('Modules not matching, performing partial update')
            new_state = {k: v for k, v in new_state.items() if k in old_state}
            old_state.update(new_state)
            new_state = old_state
        self.load_state_dict(new_state)

        log.info(f'Loaded weights from {weights_file}')