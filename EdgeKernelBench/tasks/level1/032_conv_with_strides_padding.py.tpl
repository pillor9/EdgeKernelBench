# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv_with_strides_padding.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_032_conv_with_strides_padding',
 'level': 1,
 'name': 'conv_with_strides_padding',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Convolution/Conv_with_strides_padding.py',
                          'symbol': 'Conv_with_strides_padding'}],
 'summary': 'F.conv2d(x, W, bias=None, stride=(2,2), padding=(1,1), dilation=(1,1), groups=1)'}
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Model(nn.Module):

    def __init__(self, strides: Tuple[int, int]=(2, 2), pads: Tuple[int, int, int, int]=(1, 1, 1, 1), dilations: Tuple[int, int]=(1, 1), group: int=1, kernel_shape: Tuple[int, int]=(3, 3)):
        super().__init__()
        self.strides = strides
        self.pads = pads
        self.dilations = dilations
        self.group = group
        self.kernel_shape = kernel_shape

    def forward(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        padding = (self.pads[0], self.pads[1])
        return F.conv2d(x, W, bias=None, stride=self.strides, padding=padding, dilation=self.dilations, groups=self.group)
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_h = __EKB_PARAM_kernel_h__
kernel_w = __EKB_PARAM_kernel_w__

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    W = torch.randn(out_channels, in_channels, kernel_h, kernel_w)
    return [x, W]

def get_init_inputs():
    return [(__EKB_INPUT_get_init_inputs_001_c001__, __EKB_INPUT_get_init_inputs_001_c002__), (__EKB_INPUT_get_init_inputs_001_c003__, __EKB_INPUT_get_init_inputs_001_c004__, __EKB_INPUT_get_init_inputs_001_c005__, __EKB_INPUT_get_init_inputs_001_c006__), (__EKB_INPUT_get_init_inputs_001_c007__, __EKB_INPUT_get_init_inputs_001_c008__), __EKB_INPUT_get_init_inputs_001_c009__, (__EKB_INPUT_get_init_inputs_001_c010__, __EKB_INPUT_get_init_inputs_001_c011__)]
