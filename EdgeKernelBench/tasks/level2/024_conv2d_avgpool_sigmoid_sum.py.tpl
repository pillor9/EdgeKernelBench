# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/ScalingIntelligence--KernelBench/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv2d_avgpool_sigmoid_sum.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_024_conv2d_avgpool_sigmoid_sum',
 'level': 2,
 'name': 'conv2d_avgpool_sigmoid_sum',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'activation', 'pooling', 'reduction'],
 'semantic_references': [{'url': 'https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/65_Conv2d_AvgPool_Sigmoid_Sum.py',
                          'symbol': 'Model'}],
 'summary': 'Conv2d -> AvgPool2d -> sigmoid -> sum over C,H,W'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.avg_pool(x)
        x = torch.sigmoid(x)
        x = torch.sum(x, dim=[1, 2, 3])
        return x
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__
pool_kernel_size = __EKB_PARAM_pool_kernel_size__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
