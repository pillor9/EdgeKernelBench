# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv2d_batchnorm_scaling.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_026_conv2d_batchnorm_scaling',
 'level': 2,
 'name': 'conv2d_batchnorm_scaling',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'normalization', 'elementwise'],
 'semantic_references': [{'url': 'https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/73_Conv2d_BatchNorm_Scaling.py',
                          'symbol': 'Model'}],
 'summary': 'Conv2d -> BatchNorm2d -> scalar multiply'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = x * self.scaling_factor
        return x
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__
scaling_factor = __EKB_PARAM_scaling_factor__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]
