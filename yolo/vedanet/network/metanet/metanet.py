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
    device = None
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
                ('14_globavgpool',  nn.AdaptiveMaxPool2d(1)),       # avg pooling to size (1 x 1)
            ])

        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

        if weights_file is not None:
            self.load_weights(weights_file)


    def forward(self, x):
        # temp = [x[i] for i in range(x.shape[0])]
        # x = torch.cat(temp, 0)
        data = x
        feature = self.layers[0](data)
        weights = self.layers[1](feature)
        return weights

    def load_weights(self, weights_file, clear=False):
        """ This function will load the weights from a file.
        It also allows to load in weights file with only a part of the weights in.

        Args:
            weights_file (str): path to file
        """
        old_state = self.state_dict()
        state = torch.load(weights_file, lambda storage, loc: storage)
        self.seen = 0 if clear else state['seen']

        '''
        for key in list(state['weights'].keys()):
            if '.layer.' in key:
                log.info('Deprecated weights file found. Consider resaving your weights file before this manual intervention gets removed')
                new_key = key.replace('.layer.', '.layers.')
                state['weights'][new_key] = state['weights'].pop(key)

        new_state = state['weights']
        if new_state.keys() != old_state.keys():
            log.warn('Modules not matching, performing partial update')
            new_state = {k: v for k, v in new_state.items() if k in old_state}
            old_state.update(new_state)
            new_state = old_state
        self.load_state_dict(new_state)
        '''
        self.load_state_dict(state['weights'])

        if hasattr(self.loss, 'seen'):
            self.loss.seen = self.seen

        log.info(f'Loaded weights from {weights_file}')

    def save_weights(self, weights_file, seen=None):
        """ This function will save the weights to a file.

        Args:
            weights_file (str): path to file
            seen (int, optional): Number of images trained on; Default **self.seen**
        """
        if seen is None:
            seen = self.seen

        state = {
            'seen': seen,
            'weights': self.state_dict()
        }
        torch.save(state, weights_file)

        log.info(f'Saved weights as {weights_file}')

    # def load_weights(self, weights_file):
    #     state = torch.load(weights_file, lambda storage, loc: storage)
    #     old_state = self.state_dict()
    #
    #     for key in list(state['weights'].keys()):
    #         if 'metanet' in key:
    #             new_key = key.replace('metanet.', '')
    #             state['weights'][new_key] = state['weights'].pop(key)
    #         else:
    #             state['weights'].pop(key)
    #
    #     new_state = state['weights']
    #     if new_state.keys() != old_state.keys():
    #         log.warn('Modules not matching, performing partial update')
    #         new_state = {k: v for k, v in new_state.items() if k in old_state}
    #         old_state.update(new_state)
    #         new_state = old_state
    #     self.load_state_dict(new_state)
    #
    #     log.info(f'Loaded weights from {weights_file}')