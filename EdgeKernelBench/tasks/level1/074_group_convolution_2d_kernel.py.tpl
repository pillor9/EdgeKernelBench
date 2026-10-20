# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: group_convolution_2d_kernel.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_074_group_convolution_2d_kernel',
 'level': 1,
 'name': 'group_convolution_2d_kernel',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Convolution/Group_Convolution_2D_kernel.py',
                          'symbol': 'Group_Convolution_2D_kernel'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_ch: int=24, out_ch: int=24, kernel_size: int=5, groups: int=12, stride: int=1, padding: int=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
batch_size = __EKB_PARAM_batch_size__

def get_inputs():
    x = torch.rand(batch_size, __EKB_INPUT_get_inputs_001_c001__, __EKB_INPUT_get_inputs_001_c002__, __EKB_INPUT_get_inputs_001_c003__)
    return [x]

def get_init_inputs():
    return []
batch_size = __EKB_PARAM_batch_size__2__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__
groups = __EKB_PARAM_groups__

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, __EKB_INPUT_get_init_inputs_002_c001__, __EKB_INPUT_get_init_inputs_002_c002__]
