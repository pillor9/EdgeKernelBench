# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv2d_multiply_leakyrelu_gelu.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_037_conv2d_multiply_leakyrelu_gelu',
 'level': 2,
 'name': 'conv2d_multiply_leakyrelu_gelu',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'activation', 'elementwise'],
 'semantic_references': [{'url': 'https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/54_Conv2d_Multiply_LeakyReLU_GELU.py',
                          'symbol': 'Model'}],
 'summary': 'Conv2d -> learned multiplier -> LeakyReLU -> GELU'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        x = F.gelu(x)
        return x
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__
multiplier_shape = __EKB_PARAM_multiplier_shape__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]
