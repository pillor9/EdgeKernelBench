# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: dual_stream_placement.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_074_dual_stream_placement',
 'level': 2,
 'name': 'dual_stream_placement',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'two same-shape video streams pass through independent Conv branches and fuse by concat+Linear'}
import torch
import torch.nn as nn

class Stream(nn.Module):

    def __init__(self, channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, out_channels, 3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).mean(dim=(-2, -1))

class Model(nn.Module):

    def __init__(self, channels, out_channels, out_dim):
        super().__init__()
        self.s0 = Stream(channels, out_channels)
        self.s1 = Stream(channels, out_channels)
        self.fuse = nn.Linear(out_channels * 2, out_dim)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat((self.s0(x0), self.s1(x1)), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width), torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, out_channels, out_dim]
