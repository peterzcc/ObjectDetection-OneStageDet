import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F
from .. import layer as vn_layer

__all__ = ['WrnYolov2', 'UniWrnYolov2']
DEBUG = False

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
        self.meta_param_size = num_anchors * 6*(1024+1)
        self.register_buffer("meta_state", torch.zeros(self.num_classes, self.num_anchors*6, 1024+1))
        self.meta_state.requires_grad_(True)

    def set_meta_state(self, meta_state):
        t_device = self.meta_state.device
        self.meta_state = meta_state.detach().to(t_device)
        self.meta_state.requires_grad_(True)

    def forward(self, middle_feats):
        outputs = []
        # stage 5
        # Route : layers=-9
        stage6_reorg = self.layers[0](middle_feats[1])
        # stage 6
        stage6 = middle_feats[0]
        preout = self.layers[1](torch.cat((stage6_reorg, stage6), 1))
        out = self.meta_predict(preout)
        features = [out]
        return features

    def meta_predict(self, pre_ultimate_layer: torch.Tensor):
        batch_size = pre_ultimate_layer.shape[0]
        # (16,1024,19,19)
        # (batch, C, h, w)
        # meta_state: (num_cls, num_anchors*6*(1024+1))
        t_device = pre_ultimate_layer.device
        cls_grouped_params = self.meta_state.view(self.num_classes, self.num_anchors*6, 1024+1)

        cls_weights = cls_grouped_params[:, :, 0:-1].view(self.num_classes, self.num_anchors*6, 1024, 1, 1)
        cls_biases = cls_grouped_params[:, :, -1]
        cls_detections = [F.conv2d(pre_ultimate_layer, cls_weights[cls], cls_biases[cls], 1, 0, 1, 1)
                          for cls in range(self.num_classes)]
        stacked_detections = torch.stack(cls_detections, dim=1)
        result = stacked_detections.view(batch_size*self.num_classes, self.num_anchors*6, *pre_ultimate_layer.shape[-2:])
        return result


class UniWrnYolov2(nn.Module):
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
            ]
        self.pred_input_size = 512
        preout_layer = \
            OrderedDict([
                (f'4_convbatch',    vn_layer.Conv2dBatchLeaky(1024, num_anchors*self.pred_input_size, 1, 1)),
            ])
        layer_list.append(preout_layer)
        # uni_predictor = \
        #     OrderedDict([
        #         ('5_conv',         nn.Conv2d(self.pred_input_size, (5+num_classes), 1, 1, 0)),
        #         ])
        # layer_list.append(uni_predictor)
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])
        self.meta_param_size = 6*(self.pred_input_size+1)
        self.register_buffer("meta_state", torch.zeros(self.num_classes, self.meta_param_size,1,1))
        # self.meta_state.requires_grad_(True)

    def set_meta_state(self, meta_state):
        t_device = self.meta_state.device
        self.meta_state = meta_state.to(t_device) #.detach()
        # self.meta_state.requires_grad_(True)

    def forward(self, middle_feats):
        outputs = []
        # stage 5
        # Route : layers=-9
        stage6_reorg = self.layers[0](middle_feats[1])
        # stage 6
        stage6 = middle_feats[0]
        # Route : layers=-1, -4
        feature_layer = self.layers[1](torch.cat((stage6_reorg, stage6), 1))
        if DEBUG: assert not torch.isnan(feature_layer).any()
        anchor_aggregated = self.layers[2](feature_layer)
        if DEBUG: assert not torch.isnan(anchor_aggregated).any()
        b, _, h, w = anchor_aggregated.shape
        anchor_separated = anchor_aggregated.view(b, self.num_anchors, self.pred_input_size, h, w)
        anchor_in_batch = anchor_separated.view(b*self.num_anchors, self.pred_input_size, h, w).contiguous()
        out_anchor_in_batch = self.meta_predict(anchor_in_batch)
        out = out_anchor_in_batch.view(b*self.num_classes, self.num_anchors*6, h, w).contiguous()
        features = [out]
        return features

    def meta_predict(self, pre_ultimate_layer: torch.Tensor):
        batch_size = pre_ultimate_layer.shape[0]
        # (16,1024,19,19)
        # (batch, C, h, w)
        # meta_state: (num_cls, num_anchors*6*(1024+1))
        t_device = pre_ultimate_layer.device
        cls_grouped_params = self.meta_state.view(self.num_classes, 6, self.pred_input_size+1)

        cls_weights = 0.1*cls_grouped_params[:, :, 0:-1].view(self.num_classes, 6, self.pred_input_size, 1, 1)
        cls_biases = cls_grouped_params[:, :, -1]
        if DEBUG: assert not torch.isnan(pre_ultimate_layer).any()
        cls_detections = [F.conv2d(pre_ultimate_layer, cls_weights[cls], cls_biases[cls], 1, 0, 1, 1)
                          for cls in range(self.num_classes)]
        stacked_detections = torch.stack(cls_detections, dim=1)
        result = stacked_detections.view(batch_size*self.num_classes, 6, *pre_ultimate_layer.shape[-2:])
        if DEBUG: assert not torch.isnan(result).any()
        return result

