# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv2d_subtract_hardswish_maxpool_mish_variant_2.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_042_conv2d_subtract_hardswish_maxpool_mish_variant_2',
 'level': 2,
 'name': 'conv2d_subtract_hardswish_maxpool_mish_variant_2',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'activation', 'pooling', 'elementwise'],
 'semantic_references': [{'url': 'https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/35_Conv2d_Subtract_HardSwish_MaxPool_Mish_jetson.py',
                          'symbol': 'Model'}],
 'summary': 'Conv2d -> subtract scalar -> HardSwish -> MaxPool2d -> Mish'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x - self.subtract_value
        x = F.hardswish(x)
        x = self.pool(x)
        x = x * torch.tanh(F.softplus(x))
        return x
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__
subtract_value = __EKB_PARAM_subtract_value__
pool_kernel_size = __EKB_PARAM_pool_kernel_size__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]
