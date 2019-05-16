#
#   MetaNet model
#



import os
from collections import OrderedDict
import torch
import torch.nn as nn
import logging as log

from .. import layer as vn_layer

__all__ = ['Paramnet']


class Paramnet(nn.Module):
    """ `Metanet`_ implementation with pytorch.

    Args:
        weights_file (str, optional): Path to the saved weights; Default **None**
        input_channels (Number, optional): Number of input channels; Default **3**

    Attributes:
        self.loss (fn): loss function. Usually this is :class:`~lightnet.network.RegionLoss`
        self.postprocess (fn): Postprocessing function. By default this is :class:`~lightnet.data.GetBoundingBoxes`

    """
    device = None

    def __init__(self, num_classes=20, weights_file=None, use_dummy_reweight=False,
                 num_anchors=1, meta_param_size=5*6*1025):
        """ Network initialisation """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.param_size = meta_param_size
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
                ('13_convbatch',    nn.Conv2d(1024, self.param_size, 1, 1, 0)),
                ]),

            OrderedDict([
                ('14_globavgpool',  nn.AdaptiveMaxPool2d(1)),       # avg pooling to size (1 x 1)
            ])

        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])
        self.use_dummy_reweight = use_dummy_reweight
        self.dummy_reweight = None
        if weights_file is not None:
            self.load_weights(weights_file)

    def generate_dummy_reweight(self, device, nf=1024):
        raise NotImplementedError

    def get_dummy_reweight(self, device):
        if self.dummy_reweight is None:
            self.dummy_reweight = self.generate_dummy_reweight(device)
        return self.dummy_reweight

    def forward(self, x):
        if self.use_dummy_reweight:
            raise NotImplementedError
        data = x
        feature = self.layers[0](data)
        weights = self.layers[1](feature)
        # k_shot = int(weights.shape[0]/self.num_classes)
        # weights_batch = weights.view(k_shot, self.num_classes, *weights.shape[1:])
        # weights_aggregated = torch.mean(weights_batch, dim=0)
        return weights

    def load_weights(self, weights_file, clear=False):
        """ This function will load the weights from a file.
        It also allows to load in weights file with only a part of the weights in.

        Args:
            weights_file (str): path to file
        """
        old_state = self.state_dict()
        state = torch.load(weights_file, lambda storage, loc: storage)
        self.seen = 0

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

        log.info(f'Loaded weights from {weights_file}')

    def save_weights(self, weights_file, seen=None):
        """ This function will save the weights to a file.

        Args:
            weights_file (str): path to file
            seen (int, optional): Number of images trained on; Default **self.seen**
        """


        state = {
            'weights': self.state_dict()
        }
        torch.save(state, weights_file)

        log.info(f'Saved weights as {weights_file}')
