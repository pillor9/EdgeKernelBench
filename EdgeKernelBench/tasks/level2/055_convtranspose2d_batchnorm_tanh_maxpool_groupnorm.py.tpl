# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/ScalingIntelligence--KernelBench/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: convtranspose2d_batchnorm_tanh_maxpool_groupnorm.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_055_convtranspose2d_batchnorm_tanh_maxpool_groupnorm',
 'level': 2,
 'name': 'convtranspose2d_batchnorm_tanh_maxpool_groupnorm',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'pooling', 'tensor_transform'],
 'semantic_references': [{'url': 'https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm.py',
                          'symbol': 'Model'}],
 'summary': 'ConvTranspose2d -> BatchNorm2d -> Tanh -> MaxPool2d -> GroupNorm'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.tanh(x)
        x = self.max_pool(x)
        x = self.group_norm(x)
        return x
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__
stride = __EKB_PARAM_stride__
padding = __EKB_PARAM_padding__
groups = __EKB_PARAM_groups__
num_groups = __EKB_PARAM_num_groups__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]
