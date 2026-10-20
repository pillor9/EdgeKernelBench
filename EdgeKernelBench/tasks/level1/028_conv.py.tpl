# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_028_conv',
 'level': 1,
 'name': 'conv',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'type_conversion', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Convolution/Conv.py',
                          'symbol': 'Conv'}],
 'summary': 'source Conv wrapper with 2D example inputs; export graph is conv2d with bias'}
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class Model(nn.Module):

    def __init__(self, group: int=1, pads: List[int]=None, kernel_shape: List[int]=None, strides: List[int]=None, auto_pad: str='NOTSET', dilations: List[int]=None):
        super().__init__()
        self.group = group
        self.pads = pads if pads is not None else []
        self.kernel_shape = kernel_shape if kernel_shape is not None else []
        self.strides = strides if strides is not None else []
        self.auto_pad = auto_pad
        self.dilations = dilations if dilations is not None else []

    def forward(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]=None) -> torch.Tensor:
        conv_dim = w.ndim - 2
        strides = self.strides if self.strides else [1] * conv_dim
        dilations = self.dilations if self.dilations else [1] * conv_dim
        pads = self.pads if self.pads else [0] * (conv_dim * 2)
        if len(pads) >= conv_dim:
            padding = tuple((int(p) for p in pads[:conv_dim]))
        else:
            padding = 0
        if conv_dim == 1:
            return F.conv1d(x, w, b, stride=tuple((int(s) for s in strides)), padding=padding, dilation=tuple((int(d) for d in dilations)), groups=self.group)
        if conv_dim == 2:
            return F.conv2d(x, w, b, stride=tuple((int(s) for s in strides)), padding=padding, dilation=tuple((int(d) for d in dilations)), groups=self.group)
        if conv_dim == 3:
            return F.conv3d(x, w, b, stride=tuple((int(s) for s in strides)), padding=padding, dilation=tuple((int(d) for d in dilations)), groups=self.group)
        raise ValueError(f'Unsupported convolution dimension: {conv_dim}')
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__
groups = __EKB_PARAM_groups__
strides = __EKB_PARAM_strides__
pads = __EKB_PARAM_pads__
dilations = __EKB_PARAM_dilations__
auto_pad = __EKB_PARAM_auto_pad__

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
    b = torch.randn(out_channels)
    return [x, w, b]

def get_init_inputs():
    return [groups, pads, [], strides, auto_pad, dilations]
