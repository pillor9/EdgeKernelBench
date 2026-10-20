# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: temporal_conv_point_mlp.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_155_temporal_conv_point_mlp',
 'level': 2,
 'name': 'temporal_conv_point_mlp',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'temporal video Conv3d branch and point MLP branch are independent until concat+Linear fusion'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, point_dim, mid_channels, out_dim):
        super().__init__()
        self.temporal = nn.Sequential(nn.Conv3d(channels, mid_channels, 3, stride=(1, 2, 2), padding=1), nn.BatchNorm3d(mid_channels), nn.ReLU(), nn.Conv3d(mid_channels, mid_channels, 3, stride=(1, 2, 2), padding=1), nn.BatchNorm3d(mid_channels), nn.ReLU())
        self.point = nn.Sequential(nn.Linear(point_dim, mid_channels), nn.ReLU(), nn.Linear(mid_channels, mid_channels), nn.ReLU())
        self.fuse = nn.Linear(mid_channels * 2, out_dim)

    def forward(self, video: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        video_feat = self.temporal(video).mean(dim=(-3, -2, -1))
        point_feat = self.point(points).mean(dim=1)
        return self.fuse(torch.cat((video_feat, point_feat), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
depth = __EKB_PARAM_depth__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
seq_len = __EKB_PARAM_seq_len__
mid_channels = __EKB_PARAM_mid_channels__
out_dim = __EKB_PARAM_out_dim__
point_dim = __EKB_PARAM_point_dim__

def get_inputs():
    return [torch.rand(batch_size, channels, depth, height // __EKB_INPUT_get_inputs_001_c001__, width // __EKB_INPUT_get_inputs_001_c002__), torch.rand(batch_size, seq_len, point_dim)]

def get_init_inputs():
    return [channels, point_dim, mid_channels, out_dim]
