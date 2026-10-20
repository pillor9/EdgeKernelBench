# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: four_stream_conv_variant.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_084_four_stream_conv_variant',
 'level': 2,
 'name': 'four_stream_conv_variant',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'tensor_generation',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'four independent Conv branches process four streams and fuse by concat+Linear'}
import torch
import torch.nn as nn

class Stream(nn.Module):

    def __init__(self, channels, mid_channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, mid_channels, 3, stride=2, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU(), nn.Conv2d(mid_channels, mid_channels, 3, stride=2, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).mean(dim=(-2, -1))

class Model(nn.Module):

    def __init__(self, channels, mid_channels, out_dim):
        super().__init__()
        self.s0 = Stream(channels, mid_channels)
        self.s1 = Stream(channels, mid_channels)
        self.s2 = Stream(channels, mid_channels)
        self.s3 = Stream(channels, mid_channels)
        self.fuse = nn.Linear(mid_channels * 4, out_dim)

    def forward(self, x0, x1, x2, x3):
        return self.fuse(torch.cat((self.s0(x0), self.s1(x1), self.s2(x2), self.s3(x3)), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
mid_channels = __EKB_PARAM_mid_channels__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width) for _ in range(__EKB_INPUT_get_inputs_001_c001__)]

def get_init_inputs():
    return [channels, mid_channels, out_dim]
