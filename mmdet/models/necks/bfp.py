# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import NonLocal2d
from mmcv.runner import BaseModule
from ..builder import NECKS
from mmcv.ops import DeformConv2dPack
import torch.nn as nn
import torch
import numpy as np

def dwt_init(x):
 
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
 
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
 
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    
 
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
 
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    
    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False
        self.conv=torch.nn.Conv2d(1024,256,1)
 
    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        y = torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

        return self.conv(y)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False
        self.conv=torch.nn.Conv2d(64,256,1)
 
    def forward(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
        out_batch, out_channel, out_height, out_width = in_batch, int(
            in_channel / (r ** 2)), r * in_height, r * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    
 
        h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
 
        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    
        return self.conv(h)
 
@NECKS.register_module()
class BFP(BaseModule):

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=2,
                 refine_type=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(BFP, self).__init__(init_cfg)
        assert refine_type in [None, 'conv', 'non_local', 'dcn_v2']

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dwt=DWT()
        self.iwt=IWT()

        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level < self.num_levels

        if self.refine_type == 'conv':
            self.refine = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2d(
                self.in_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'dcn_v2':  
            self.refine = DeformConv2dPack(
                      self.in_channels,
                      self.in_channels,
                      3,
                      stride=1,
                      groups=1,
                      bias=None,
                      padding=1,
                      )
        


    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == self.num_levels
        # print('inpurts',len(inputs))

        # step 1: gather multi-level features by resize and average
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i == 0:
                
                out = self.dwt(inputs[i])
                gathered=self.dwt(out)

                # print(gathered.shape)
            elif i ==1:
                
                gathered=self.dwt(inputs[i])
                
                # print(gathered.shape)
            
            elif i ==2:
                gathered=inputs[i]
                
                # print(gathered.shape)
            else:
                gathered = F.interpolate(inputs[i], size=gather_size, mode='nearest')
                # gathered=self.iwt(inputs[i])
            
                # print(gathered.shape)
            feats.append(gathered)

        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i == 0:
                out = self.iwt(bsf)
                residual=self.iwt(out)
            elif i ==1:
                residual=self.iwt(bsf)
            elif i ==2:
                residual=bsf
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + inputs[i])

        return tuple(outs)
    

    # def forward(self, inputs):
    #     """Forward function."""
    #     assert len(inputs) == self.num_levels

    #     # step 1: gather multi-level features by resize and average
    #     feats = []
    #     gather_size = inputs[self.refine_level].size()[2:]
    #     for i in range(self.num_levels):
    #         if i < self.refine_level:
    #             gathered = F.adaptive_max_pool2d(
    #                 inputs[i], output_size=gather_size)
    #         else:
    #             gathered = F.interpolate(
    #                 inputs[i], size=gather_size, mode='nearest')
    #         feats.append(gathered)

    #     bsf = sum(feats) / len(feats)

    #     # step 2: refine gathered features
    #     if self.refine_type is not None:
    #         bsf = self.refine(bsf)

    #     # step 3: scatter refined features to multi-levels by a residual path
    #     outs = []
    #     for i in range(self.num_levels):
    #         out_size = inputs[i].size()[2:]
    #         if i < self.refine_level:
    #             residual = F.interpolate(bsf, size=out_size, mode='nearest')
    #         else:
    #             residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
    #         outs.append(residual + inputs[i])

    #     return tuple(outs)
