# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv2d_relu_hardswish.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_039_conv2d_relu_hardswish',
 'level': 2,
 'name': 'conv2d_relu_hardswish',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'activation', 'elementwise'],
 'semantic_references': [{'url': 'https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/57_Conv2d_ReLU_HardSwish.py',
                          'symbol': 'Model'}],
 'summary': 'Conv2d -> ReLU -> x * clamp((x + 3) / 6, 0, 1)'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.relu(x)
        x = x * torch.clamp((x + 3) / 6, 0, 1)
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
