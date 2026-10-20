# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: pointpillars_backbone.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_027_pointpillars_backbone',
 'level': 3,
 'name': 'pointpillars_backbone',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/voxel_encoders/pillar_encoder.py'}],
 'summary': 'Source: '
            'https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/voxel_encoders/pillar_encoder.py '
            '(PillarFeatureNet).',
 'adaptation_notes': ['static PillarVFE max aggregation, fixed ordered-pillar scatter represented by exact '
                      'reshape, and two-stage 2D backbone.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, features: int, channels: int, grid_h: int, grid_w: int):
        super().__init__()
        self.gh = grid_h
        self.gw = grid_w
        self.pfn = nn.Linear(features, channels)
        self.conv1 = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(channels, 2 * channels, 3, stride=2, padding=1), nn.BatchNorm2d(2 * channels), nn.ReLU(), nn.Conv2d(2 * channels, 2 * channels, 3, padding=1), nn.ReLU())

    def forward(self, pillars: torch.Tensor) -> torch.Tensor:
        b, p, n, _ = pillars.shape
        x = F.relu(self.pfn(pillars))
        x = torch.max(x, dim=2)[0]
        x = x.transpose(1, 2).reshape(b, -1, self.gh, self.gw)
        return self.conv2(self.conv1(x))
batch_size = __EKB_PARAM_batch_size__
grid_h = __EKB_PARAM_grid_h__
grid_w = __EKB_PARAM_grid_w__
points = __EKB_PARAM_points__
features = __EKB_PARAM_features__
channels = __EKB_PARAM_channels__

def get_inputs():
    return [torch.randn(batch_size, grid_h * grid_w, points, features)]

def get_init_inputs():
    return [features, channels, grid_h, grid_w]
