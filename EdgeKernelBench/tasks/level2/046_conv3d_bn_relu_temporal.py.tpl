# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv3d_bn_relu_temporal.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_046_conv3d_bn_relu_temporal',
 'level': 2,
 'name': 'conv3d_bn_relu_temporal',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution', 'normalization', 'activation'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py',
                          'symbol': 'Conv3DSimple'}],
 'summary': 'torchvision video ResNet temporal unit: Conv3DSimple 3x3x3 -> BatchNorm3d -> ReLU'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False), nn.BatchNorm3d(out_channels), nn.ReLU(inplace=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
depth = __EKB_PARAM_depth__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, depth, height, width)]

def get_init_inputs():
    return [channels, out_channels]
