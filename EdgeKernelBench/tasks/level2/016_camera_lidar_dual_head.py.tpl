# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: camera_lidar_dual_head.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_016_camera_lidar_dual_head',
 'level': 2,
 'name': 'camera_lidar_dual_head',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'Concatenate the outputs of independent camera and lidar prediction heads.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, point_dim, mid_channels, out_dim):
        super().__init__()
        self.camera = nn.Sequential(nn.Conv2d(channels, mid_channels, 3, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU(), nn.Conv2d(mid_channels, mid_channels, 3, stride=2, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU())
        self.lidar = nn.Sequential(nn.Conv1d(point_dim, mid_channels, 1), nn.BatchNorm1d(mid_channels), nn.ReLU(), nn.Conv1d(mid_channels, mid_channels, 1), nn.BatchNorm1d(mid_channels), nn.ReLU())
        self.camera_head = nn.Linear(mid_channels, out_dim)
        self.lidar_head = nn.Linear(mid_channels, out_dim)

    def forward(self, image: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        cam = self.camera_head(self.camera(image).mean(dim=(-2, -1)))
        lidar = self.lidar_head(self.lidar(points).max(dim=-1).values)
        return torch.cat((cam, lidar), dim=1)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
mid_channels = __EKB_PARAM_mid_channels__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
seq_len = __EKB_PARAM_seq_len__
point_dim = __EKB_PARAM_point_dim__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width), torch.rand(batch_size, point_dim, seq_len)]

def get_init_inputs():
    return [channels, point_dim, mid_channels, out_dim]
