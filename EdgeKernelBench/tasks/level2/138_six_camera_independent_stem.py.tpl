# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: six_camera_independent_stem.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_138_six_camera_independent_stem',
 'level': 2,
 'name': 'six_camera_independent_stem',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'tensor_generation',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'six camera tensors pass through independent Conv-BN-ReLU stems and are fused by concat+Linear'}
import torch
import torch.nn as nn

class CameraStem(nn.Module):

    def __init__(self, channels, mid_channels):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(channels, mid_channels, 3, stride=2, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU(), nn.Conv2d(mid_channels, mid_channels, 3, stride=2, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x).mean(dim=(-2, -1))

class Model(nn.Module):

    def __init__(self, channels, mid_channels, out_dim):
        super().__init__()
        self.cam0 = CameraStem(channels, mid_channels)
        self.cam1 = CameraStem(channels, mid_channels)
        self.cam2 = CameraStem(channels, mid_channels)
        self.cam3 = CameraStem(channels, mid_channels)
        self.cam4 = CameraStem(channels, mid_channels)
        self.cam5 = CameraStem(channels, mid_channels)
        self.fuse = nn.Linear(mid_channels * 6, out_dim)

    def forward(self, c0, c1, c2, c3, c4, c5):
        feats = (self.cam0(c0), self.cam1(c1), self.cam2(c2), self.cam3(c3), self.cam4(c4), self.cam5(c5))
        return self.fuse(torch.cat(feats, dim=1))
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
