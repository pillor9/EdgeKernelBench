# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: image_geometry_transform.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_098_image_geometry_transform',
 'level': 2,
 'name': 'image_geometry_transform',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'image Conv2d branch runs independently of a rigid 3D point transform plus MLP branch, then '
            'concat+Linear fuse'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, mid_channels, out_dim):
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(channels, mid_channels, 3, stride=2, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU(), nn.Conv2d(mid_channels, mid_channels, 3, stride=2, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU())
        self.point_mlp = nn.Sequential(nn.Linear(3, mid_channels), nn.ReLU(), nn.Linear(mid_channels, mid_channels), nn.ReLU())
        self.fuse = nn.Linear(mid_channels * 2, out_dim)

    def forward(self, image: torch.Tensor, points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        img_feat = self.image(image).mean(dim=(-2, -1))
        rot = transform[:, :, :3]
        trans = transform[:, :, 3]
        pts = torch.bmm(points, rot.transpose(1, 2)) + trans.unsqueeze(1)
        geo_feat = self.point_mlp(pts).mean(dim=1)
        return self.fuse(torch.cat((img_feat, geo_feat), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
mid_channels = __EKB_PARAM_mid_channels__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
seq_len = __EKB_PARAM_seq_len__

def get_inputs():
    image = torch.rand(batch_size, channels, height, width)
    points = torch.rand(batch_size, seq_len, __EKB_INPUT_get_inputs_001_c001__)
    transform = torch.rand(batch_size, __EKB_INPUT_get_inputs_001_c002__, __EKB_INPUT_get_inputs_001_c003__)
    return [image, points, transform]

def get_init_inputs():
    return [channels, mid_channels, out_dim]
