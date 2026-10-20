# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: convtranspose_generator.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_060_convtranspose_generator',
 'level': 2,
 'name': 'convtranspose_generator',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'normalization', 'activation'],
 'semantic_references': [{'url': 'https://github.com/pytorch/examples/blob/main/dcgan/main.py'}],
 'summary': 'Source: https://github.com/pytorch/examples/blob/main/dcgan/main.py (Generator).',
 'adaptation_notes': ['DCGAN generator cell with ConvTranspose2d, BatchNorm2d, and ReLU.']}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(channels, out_channels, 4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.deconv(x)))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, out_channels]
