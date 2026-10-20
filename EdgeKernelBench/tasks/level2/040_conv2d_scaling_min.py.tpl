# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv2d_scaling_min.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_040_conv2d_scaling_min',
 'level': 2,
 'name': 'conv2d_scaling_min',
 'scenario_tags': ['mobile_personal', 'smart_aiot'],
 'operator_tags': ['convolution', 'reduction', 'elementwise'],
 'semantic_references': [{'url': 'https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/32_Conv2d_Scaling_Min.py',
                          'symbol': 'Model'}],
 'summary': 'Conv2d -> scalar multiply -> channel min(dim=1, keepdim=True)'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x * self.scale_factor
        x = -torch.max(-x, dim=1, keepdim=True)[0]
        return x
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__
scale_factor = __EKB_PARAM_scale_factor__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
