# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: convtranspose_2d_asymmetric_input_asymmetric_kernel.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_034_convtranspose_2d_asymmetric_input_asymmetric_kernel',
 'level': 1,
 'name': 'convtranspose_2d_asymmetric_input_asymmetric_kernel',
 'scenario_tags': ['mobile_personal', 'smart_aiot'],
 'operator_tags': ['convolution'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Convolution/ConvTranspose_2D__asymmetric_input__asymmetric_kernel.py',
                          'symbol': 'ConvTranspose_2D__asymmetric_input__asymmetric_kernel'}],
 'summary': 'nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3,5), bias=False)(x)'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple=(1, 1), padding: tuple=(0, 0), output_padding: tuple=(0, 0), dilation: tuple=(1, 1), groups: int=1, bias: bool=False):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose2d(x)
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
kernel_size = __EKB_PARAM_kernel_size__
height_in = __EKB_PARAM_height_in__
width_in = __EKB_PARAM_width_in__

def get_inputs():
    x = torch.rand(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
