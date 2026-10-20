# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv3d_variant_2.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_030_conv3d_variant_2',
 'level': 1,
 'name': 'conv3d_variant_2',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py',
                          'symbol': 'Conv3DSimple'}],
 'summary': 'torchvision video ResNet Conv3DSimple is nn.Conv3d kernel_size=(3,3,3), stride=1, padding=1, '
            'bias=False'}
import torch
import torch.nn as nn

class Conv3DSimple(nn.Conv3d):

    def __init__(self, in_planes: int, out_planes: int, stride: int=1, padding: int=1):
        super().__init__(in_channels=in_planes, out_channels=out_planes, kernel_size=(3, 3, 3), stride=stride, padding=padding, bias=False)

class Model(nn.Module):

    def __init__(self, in_planes: int, out_planes: int):
        super().__init__()
        self.conv = Conv3DSimple(in_planes, out_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
batch_size = __EKB_PARAM_batch_size__
in_planes = __EKB_PARAM_in_planes__
out_planes = __EKB_PARAM_out_planes__
depth = __EKB_PARAM_depth__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, in_planes, depth, height, width)]

def get_init_inputs():
    return [in_planes, out_planes]
