# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv2d_min_tanh_tanh.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_035_conv2d_min_tanh_tanh',
 'level': 2,
 'name': 'conv2d_min_tanh_tanh',
 'scenario_tags': ['mobile_personal', 'smart_aiot'],
 'operator_tags': ['convolution', 'activation', 'reduction', 'elementwise'],
 'semantic_references': [{'url': 'https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/25_Conv2d_Min_Tanh_Tanh.py',
                          'symbol': 'Model'}],
 'summary': 'Conv2d -> channel min(dim=1, keepdim=True) -> Tanh -> Tanh'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = -torch.max(-x, dim=1, keepdim=True)[0]
        x = torch.tanh(x)
        x = torch.tanh(x)
        return x
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
