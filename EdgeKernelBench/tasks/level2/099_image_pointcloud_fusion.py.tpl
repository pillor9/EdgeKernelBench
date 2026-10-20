# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: image_pointcloud_fusion.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_099_image_pointcloud_fusion',
 'level': 2,
 'name': 'image_pointcloud_fusion',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'independent image Conv2d branch and point-cloud Conv1d MLP+max branch fused by concat+Linear'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, img_channels, point_dim, mid_channels, out_dim):
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(img_channels, mid_channels, 3, stride=2, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU(), nn.Conv2d(mid_channels, mid_channels, 3, stride=2, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU())
        self.point = nn.Sequential(nn.Conv1d(point_dim, mid_channels, 1), nn.BatchNorm1d(mid_channels), nn.ReLU(), nn.Conv1d(mid_channels, mid_channels, 1), nn.BatchNorm1d(mid_channels), nn.ReLU())
        self.fuse = nn.Linear(mid_channels * 2, out_dim)

    def forward(self, image: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        img_feat = self.image(image).mean(dim=(-2, -1))
        point_feat = self.point(points).max(dim=-1).values
        return self.fuse(torch.cat((img_feat, point_feat), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
mid_channels = __EKB_PARAM_mid_channels__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
seq_len = __EKB_PARAM_seq_len__
point_dim = __EKB_PARAM_point_dim__

def get_inputs():
    image = torch.rand(batch_size, channels, height, width)
    points = torch.rand(batch_size, point_dim, seq_len)
    return [image, points]

def get_init_inputs():
    return [channels, point_dim, mid_channels, out_dim]
