# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: patch_embed_geometry_variant.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_119_patch_embed_geometry_variant',
 'level': 2,
 'name': 'patch_embed_geometry_variant',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'embedding',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'patch embedding branch and rigid point transform MLP branch are independent until concat+Linear '
            'fusion'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, point_dim, mid_channels, out_dim, patch_size):
        super().__init__()
        self.patch = nn.Conv2d(channels, out_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(out_dim)
        self.point = nn.Sequential(nn.Linear(point_dim, mid_channels), nn.ReLU(), nn.Linear(mid_channels, out_dim), nn.ReLU())
        self.fuse = nn.Linear(out_dim * 2, out_dim)

    def forward(self, image: torch.Tensor, points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        patches = self.patch(image).flatten(2).transpose(1, 2)
        patch_feat = self.norm(patches).mean(dim=1)
        rot = transform[:, :, :3]
        trans = transform[:, :, 3]
        pts = torch.bmm(points, rot.transpose(1, 2)) + trans.unsqueeze(1)
        point_feat = self.point(pts).mean(dim=1)
        return self.fuse(torch.cat((patch_feat, point_feat), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
mid_channels = __EKB_PARAM_mid_channels__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
seq_len = __EKB_PARAM_seq_len__
patch_size = __EKB_PARAM_patch_size__
point_dim = __EKB_PARAM_point_dim__

def get_inputs():
    image = torch.rand(batch_size, channels, height, width)
    points = torch.rand(batch_size, seq_len, point_dim)
    transform = torch.rand(batch_size, __EKB_INPUT_get_inputs_001_c001__, __EKB_INPUT_get_inputs_001_c002__)
    return [image, points, transform]

def get_init_inputs():
    return [channels, point_dim, mid_channels, out_dim, patch_size]
