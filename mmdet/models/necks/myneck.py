import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmdet.models.necks import FPN
from mmdet.models.necks.bfp import BFP
from mmdet.models.necks.dyhead import DyHead
from ..builder import NECKS

class GCNet(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(GCNet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels // reduction, out_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
class FeatureFusion(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(FeatureFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels1, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels2, out_channels, kernel_size=1)
        self.gcnet = GCNet(out_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x = x1 + x2
        x = self.gcnet(x)
        return x
@NECKS.register_module()
class MyNeck1(nn.Module):
    def __init__(self):
        super(MyNeck1, self).__init__()
        self.fpn = FPN(in_channels=[256, 512, 1024, 2048], out_channels=256, start_level=1, add_extra_convs='on_output', num_outs=5, relu_before_extra_convs=True)
        self.bfp = BFP(in_channels=256, num_levels=5, refine_level=2, refine_type='dcn_v2')
        self.dyhead = DyHead(in_channels=256, out_channels=256, num_blocks=6)
    def forward(self, x):
        fpn_outs = self.fpn(x)
        bfp_outs = self.bfp(fpn_outs)
        dyhead_outs = self.dyhead(fpn_outs)
        fusion = FeatureFusion(in_channels1=64, in_channels2=128, out_channels=128)
        
        fusion_outs = []
        for i in range(len(bfp_outs)):
            fusion=fusion.to('cuda')
            fusion_feature = fusion(bfp_outs[i], dyhead_outs[i])
            
            fusion_outs.append(fusion_feature)
        return tuple(fusion_outs)
    









