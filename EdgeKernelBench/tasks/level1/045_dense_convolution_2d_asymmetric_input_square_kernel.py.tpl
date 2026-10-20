# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: dense_convolution_2d_asymmetric_input_square_kernel.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_045_dense_convolution_2d_asymmetric_input_square_kernel',
 'level': 1,
 'name': 'dense_convolution_2d_asymmetric_input_square_kernel',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Convolution/Dense_Convolution_2D__asymmetric_input__square_kernel.py',
                          'symbol': 'Dense_Convolution_2D__asymmetric_input__square_kernel'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, bias: bool=False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)
batch_size = __EKB_PARAM_batch_size__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
kernel_size = __EKB_PARAM_kernel_size__

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
