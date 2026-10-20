# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: bev_token_pointnet_fusion.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_007_bev_token_pointnet_fusion',
 'level': 2,
 'name': 'bev_token_pointnet_fusion',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'BEV token MLP branch and PointNet-style point branch are independent until concat+Linear fusion'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, dim, point_dim, mid_channels, out_dim):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, out_dim), nn.GELU(), nn.LayerNorm(out_dim))
        self.point = nn.Sequential(nn.Conv1d(point_dim, mid_channels, 1), nn.BatchNorm1d(mid_channels), nn.ReLU(), nn.Conv1d(mid_channels, out_dim, 1), nn.BatchNorm1d(out_dim), nn.ReLU())
        self.fuse = nn.Linear(out_dim * 2, out_dim)

    def forward(self, tokens: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        token_feat = self.token(tokens).mean(dim=1)
        point_feat = self.point(points).max(dim=-1).values
        return self.fuse(torch.cat((token_feat, point_feat), dim=1))
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__
mid_channels = __EKB_PARAM_mid_channels__
out_dim = __EKB_PARAM_out_dim__
point_dim = __EKB_PARAM_point_dim__

def get_inputs():
    tokens = torch.rand(batch_size, seq_len, dim)
    points = torch.rand(batch_size, point_dim, seq_len)
    return [tokens, points]

def get_init_inputs():
    return [dim, point_dim, mid_channels, out_dim]
