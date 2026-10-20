# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: gcnet_context.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_087_gcnet_context',
 'level': 2,
 'name': 'gcnet_context',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/xvjiarui/GCNet'}],
 'summary': 'Source: https://github.com/xvjiarui/GCNet (ContextBlock).',
 'adaptation_notes': ['1x1 attention mask, spatial softmax pooling, bottleneck transform, broadcast add.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        self.mask = nn.Conv2d(channels, 1, 1)
        self.transform1 = nn.Conv2d(channels, reduction, 1)
        self.transform2 = nn.Conv2d(reduction, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        weights = F.softmax(self.mask(x).reshape(b, 1, h * w), dim=-1)
        context = torch.bmm(x.reshape(b, c, h * w), weights.transpose(1, 2)).reshape(b, c, 1, 1)
        return x + self.transform2(F.relu(self.transform1(context)))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
reduction = __EKB_PARAM_reduction__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, reduction]
