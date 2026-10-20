# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/ScalingIntelligence--KernelBench/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv2d_tanh_scaling_biasadd_max.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_045_conv2d_tanh_scaling_biasadd_max',
 'level': 2,
 'name': 'conv2d_tanh_scaling_biasadd_max',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'activation', 'pooling', 'elementwise'],
 'semantic_references': [{'url': 'https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/82_Conv2d_Tanh_Scaling_BiasAdd_Max.py',
                          'symbol': 'Model'}],
 'summary': 'Conv2d -> tanh -> scalar multiply -> bias add -> MaxPool2d'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.tanh(x)
        x = x * self.scaling_factor
        x = x + self.bias
        x = self.max_pool(x)
        return x
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__
scaling_factor = __EKB_PARAM_scaling_factor__
bias_shape = __EKB_PARAM_bias_shape__
pool_kernel_size = __EKB_PARAM_pool_kernel_size__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]
