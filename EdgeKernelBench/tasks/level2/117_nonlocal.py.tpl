# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: nonlocal.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_117_nonlocal',
 'level': 2,
 'name': 'nonlocal',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention',
                   'convolution',
                   'matrix_multiplication',
                   'activation',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_embedded_gaussian.py'}],
 'summary': 'Source: '
            'https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_embedded_gaussian.py '
            '(NONLocalBlock2D).',
 'adaptation_notes': ['embedded-Gaussian theta/phi/g 1x1 projections, spatial affinity softmax, output 1x1 '
                      'projection, and residual add.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, channels: int, inter_channels: int):
        super().__init__()
        self.theta = nn.Conv2d(channels, inter_channels, 1)
        self.phi = nn.Conv2d(channels, inter_channels, 1)
        self.g = nn.Conv2d(channels, inter_channels, 1)
        self.out = nn.Conv2d(inter_channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        theta = self.theta(x).reshape(b, -1, h * w).transpose(1, 2)
        phi = self.phi(x).reshape(b, -1, h * w)
        attention = F.softmax(torch.bmm(theta, phi), dim=-1)
        value = self.g(x).reshape(b, -1, h * w).transpose(1, 2)
        y = torch.bmm(attention, value).transpose(1, 2).reshape(b, -1, h, w)
        return x + self.out(y)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
inter_channels = __EKB_PARAM_inter_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, inter_channels]
