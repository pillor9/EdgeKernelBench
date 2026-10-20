# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: tcn_full.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_036_tcn_full',
 'level': 3,
 'name': 'tcn_full',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution', 'activation', 'recurrent', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/locuslab/TCN/blob/master/TCN/tcn.py'}],
 'summary': 'Source: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py (TemporalConvNet/TemporalBlock).',
 'adaptation_notes': ['full static four-block TCN with dilations 1/2/4/8, causal padding followed by exact '
                      'static chomp, ReLU and residual paths.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):

    def __init__(self, ci: int, co: int, d: int):
        super().__init__()
        self.chomp = 2 * d
        self.c1 = nn.Conv1d(ci, co, 3, padding=2 * d, dilation=d)
        self.c2 = nn.Conv1d(co, co, 3, padding=2 * d, dilation=d)
        self.down = nn.Conv1d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.c1(x)[..., :-self.chomp])
        y = F.relu(self.c2(y)[..., :-self.chomp])
        return F.relu(y + self.down(x))

class Model(nn.Module):

    def __init__(self, inputs: int, channels: int, outputs: int):
        super().__init__()
        self.net = nn.Sequential(Block(inputs, channels, 1), Block(channels, channels, 2), Block(channels, channels, 4), Block(channels, channels, 8))
        self.head = nn.Conv1d(channels, outputs, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))
batch_size = __EKB_PARAM_batch_size__
inputs = __EKB_PARAM_inputs__
channels = __EKB_PARAM_channels__
outputs = __EKB_PARAM_outputs__
length = __EKB_PARAM_length__

def get_inputs():
    return [torch.randn(batch_size, inputs, length)]

def get_init_inputs():
    return [inputs, channels, outputs]
