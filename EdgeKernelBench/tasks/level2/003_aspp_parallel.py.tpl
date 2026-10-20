# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: aspp_parallel.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_003_aspp_parallel',
 'level': 2,
 'name': 'aspp_parallel',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution',
                   'normalization',
                   'activation',
                   'pooling',
                   'reduction',
                   'resampling',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py',
                          'symbol': 'ASPP'}],
 'summary': 'torchvision ASPP: 1x1 branch + dilated conv branches + pooling branch -> concat -> 1x1 project'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False))

class ASPPPooling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=(-2, -1), keepdim=True)
        x = self.conv(x)
        return F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)

class Model(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False))
        self.branch2 = ASPPConv(in_channels, out_channels, dilation=12)
        self.branch3 = ASPPConv(in_channels, out_channels, dilation=24)
        self.branch4 = ASPPConv(in_channels, out_channels, dilation=36)
        self.branch5 = ASPPPooling(in_channels, out_channels)
        self.project = nn.Sequential(nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        y5 = self.branch5(x)
        return self.project(torch.cat([y1, y2, y3, y4, y5], dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, out_channels]
