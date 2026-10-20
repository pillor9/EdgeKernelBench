# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: instancenorm_residual.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_101_instancenorm_residual',
 'level': 2,
 'name': 'instancenorm_residual',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'residual', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/examples/blob/main/fast_neural_style/neural_style/transformer_net.py'}],
 'summary': 'Source: '
            'https://github.com/pytorch/examples/blob/main/fast_neural_style/neural_style/transformer_net.py '
            '(ResidualBlock).',
 'adaptation_notes': ['two Conv2d+InstanceNorm2d stages with a residual addition; ReLU is after the first '
                      'normalization only.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.norm2(self.conv2(F.relu(self.norm1(self.conv1(x)))))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels]
