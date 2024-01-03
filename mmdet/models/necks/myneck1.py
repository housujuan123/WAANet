import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmdet.models.necks import FPN
from mmdet.models.necks.bfp import BFP
from ..builder import NECKS

import torch.nn as nn
import torch.nn.functional as F

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
 
    def forward(self, x):
        return dwt_init(x)
 
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False
 
    def forward(self, x):
        return iwt_init(x)


class SAM(nn.Module):
    def __init__(self, in_channels):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x1, x2):
        batch_size, channels, height1, width1 = x1.size()
        _, _, height2, width2 = x2.size()
        conv1_1 = self.conv1(x1)
        conv1_2 = self.conv1(x2)
        conv2_1 = self.conv2(conv1_1)
        conv2_2 = self.conv2(conv1_2)
        conv3_1 = self.conv3(conv2_1)
        conv3_2 = self.conv3(conv2_2)
        avg_pool_1 = F.avg_pool2d(conv3_1, kernel_size=(height1, width1))
        max_pool_1 = F.max_pool2d(conv3_1, kernel_size=(height1, width1))
        avg_pool_2 = F.avg_pool2d(conv3_2, kernel_size=(height2, width2))
        max_pool_2 = F.max_pool2d(conv3_2, kernel_size=(height2, width2))
        out = torch.cat([avg_pool_1, max_pool_1, avg_pool_2, max_pool_2], dim=1)
        out = out.view(batch_size, -1)
        out = F.softmax(out, dim=1)
        out = out.view(batch_size, 4 * channels, 1, 1)
        out1 = conv3_1 * out[:, 0:1, :, :].expand_as(conv3_1)
        out2 = conv3_1 * out[:, 1:2, :, :].expand_as(conv3_1)
        out3 = conv3_2 * out[:, 2:3, :, :].expand_as(conv3_2)
        out4 = conv3_2 * out[:, 3:4, :, :].expand_as(conv3_2)
        out = out1 + out2 + out3 + out4
        out = self.gamma * out + x1
        return out
@NECKS.register_module()
class MyNeck(nn.Module):
    def __init__(self):
        super(MyNeck, self).__init__()
        self.fpn = FPN(in_channels=[256, 512, 1024, 2048], out_channels=256, start_level=1, add_extra_convs='on_output', num_outs=5, relu_before_extra_convs=True)
        self.bfp = BFP(in_channels=256, num_levels=5, refine_level=2, refine_type='dcn_v2')
    def forward(self, x):
        fpn_outs = self.fpn(x)
        bfp_outs = self.bfp(fpn_outs)
        fusion = SAM(256)
        
        fusion_outs = []
        for i in range(len(bfp_outs)):
            fusion=fusion.to('cuda')
            fusion_feature = fusion(bfp_outs[i], fpn_outs[i])
            
            fusion_outs.append(fusion_feature)
        return tuple(fusion_outs)
    