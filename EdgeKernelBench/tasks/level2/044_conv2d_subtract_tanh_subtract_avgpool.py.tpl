# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/ScalingIntelligence--KernelBench/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv2d_subtract_tanh_subtract_avgpool.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_044_conv2d_subtract_tanh_subtract_avgpool',
 'level': 2,
 'name': 'conv2d_subtract_tanh_subtract_avgpool',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'activation', 'pooling', 'elementwise'],
 'semantic_references': [{'url': 'https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/46_Conv2d_Subtract_Tanh_Subtract_AvgPool.py',
                          'symbol': 'Model'}],
 'summary': 'Conv2d -> subtract scalar -> tanh -> subtract scalar -> AvgPool2d'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x - self.subtract1_value
        x = torch.tanh(x)
        x = x - self.subtract2_value
        x = self.avgpool(x)
        return x
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__
subtract1_value = __EKB_PARAM_subtract1_value__
subtract2_value = __EKB_PARAM_subtract2_value__
kernel_size_pool = __EKB_PARAM_kernel_size_pool__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]
