# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv3d.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_029_conv3d',
 'level': 1,
 'name': 'conv3d',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Convolution/Conv3D.py',
                          'symbol': 'Conv3D'}],
 'summary': 'nn.Conv3d module applied to [N, C_in, D, H, W]'}
import torch
import torch.nn as nn
from typing import Tuple, Union

class Model(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int, int]], stride: Union[int, Tuple[int, int, int]]=1, padding: Union[int, Tuple[int, int, int]]=0, dilation: Union[int, Tuple[int, int, int]]=1, groups: int=1, bias: bool=True, padding_mode: str='zeros'):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3d(x)
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
depth = __EKB_PARAM_depth__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__
stride = __EKB_PARAM_stride__
padding = __EKB_PARAM_padding__
dilation = __EKB_PARAM_dilation__
groups = __EKB_PARAM_groups__
bias = __EKB_PARAM_bias__

def get_inputs():
    torch.manual_seed(__EKB_INPUT_get_inputs_001_c001__)
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias]
