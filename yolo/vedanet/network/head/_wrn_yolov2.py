import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F
from .. import layer as vn_layer

__all__ = ['WrnYolov2']


class WrnYolov2(nn.Module):
    def  __init__(self, num_anchors, num_classes, input_channels=48):
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

            # # Sequence 4 : input = sequence3 * reweight
            # OrderedDict([
            #     ('4_conv', nn.Conv2d(1024, num_anchors * 6, 1, 1, 0)),  # o, x, y, h, w, c
            # ]),
            ]

        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.reweight = None

    def forward(self, middle_feats):
        outputs = []
        # stage 5
        # Route : layers=-9
        stage6_reorg = self.layers[0](middle_feats[1])
        # stage 6
        stage6 = middle_feats[0]
        preout = self.layers[1](torch.cat((stage6_reorg, stage6), 1))
        out = self.get_reweighted_output(preout, self.reweight)
        features = [out]
        return features

    def get_reweighted_output(self, pre_ultimate_layer: torch.Tensor, meta_state):
        batch_size = pre_ultimate_layer.shape[0]
        # (16,1024,19,19)
        # (batch, C, h, w)
        # meta_state: (num_cls, num_anchors*6*(1024+1))
        cls_grouped_params = meta_state.view(self.num_classes, self.num_anchors*6, 1024+1).to(pre_ultimate_layer.device)
        cls_weights = cls_grouped_params[:, :, 0:-1].view(self.num_classes, self.num_anchors*6, 1024, 1, 1)
        cls_biases = cls_grouped_params[:, :, -1]
        cls_detections = [F.conv2d(pre_ultimate_layer, cls_weights[cls], cls_biases[cls], 1, 0, 1, 1)
                          for cls in range(self.num_classes)]
        stacked_detections = torch.stack(cls_detections, dim=1)
        result = stacked_detections.view(batch_size*self.num_classes, self.num_anchors*6, *pre_ultimate_layer.shape[-2:])
        return result

