# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: convtranspose2d_gelu_groupnorm.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_057_convtranspose2d_gelu_groupnorm',
 'level': 2,
 'name': 'convtranspose2d_gelu_groupnorm',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'tensor_transform'],
 'semantic_references': [{'url': 'https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/19_ConvTranspose2d_GELU_GroupNorm.py',
                          'symbol': 'Model'}],
 'summary': 'ConvTranspose2d -> GELU -> GroupNorm'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = F.gelu(x)
        x = self.group_norm(x)
        return x
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__
stride = __EKB_PARAM_stride__
groups = __EKB_PARAM_groups__
num_groups = __EKB_PARAM_num_groups__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]
