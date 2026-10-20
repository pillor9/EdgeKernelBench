# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: stereo_cost_volume_point_branch.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_146_stereo_cost_volume_point_branch',
 'level': 2,
 'name': 'stereo_cost_volume_point_branch',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'static shifted stereo cost-volume Conv3d branch runs independently of point-cloud Conv1d branch '
            'before fusion'}
import torch
import torch.nn as nn
import torch.nn.functional as F

def shifted_pair(left: torch.Tensor, right: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return torch.cat((left, right), dim=1)
    left_s = F.pad(left[..., :, shift:], (shift, 0, 0, 0))
    right_s = F.pad(right[..., :, :-shift], (shift, 0, 0, 0))
    return torch.cat((left_s, right_s), dim=1)

class Model(nn.Module):

    def __init__(self, channels, point_dim, mid_channels, out_dim):
        super().__init__()
        self.cost = nn.Sequential(nn.Conv3d(channels * 2, mid_channels, 3, padding=1), nn.BatchNorm3d(mid_channels), nn.ReLU())
        self.point = nn.Sequential(nn.Conv1d(point_dim, mid_channels, 1), nn.BatchNorm1d(mid_channels), nn.ReLU())
        self.fuse = nn.Linear(mid_channels * 2, out_dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        volume = torch.stack((shifted_pair(left, right, 0), shifted_pair(left, right, 1), shifted_pair(left, right, 2), shifted_pair(left, right, 3)), dim=2)
        stereo_feat = self.cost(volume).mean(dim=(-3, -2, -1))
        point_feat = self.point(points).max(dim=-1).values
        return self.fuse(torch.cat((stereo_feat, point_feat), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
mid_channels = __EKB_PARAM_mid_channels__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
seq_len = __EKB_PARAM_seq_len__
point_dim = __EKB_PARAM_point_dim__

def get_inputs():
    left = torch.rand(batch_size, channels, height // __EKB_INPUT_get_inputs_001_c001__, width // __EKB_INPUT_get_inputs_001_c002__)
    right = torch.rand(batch_size, channels, height // __EKB_INPUT_get_inputs_001_c003__, width // __EKB_INPUT_get_inputs_001_c004__)
    points = torch.rand(batch_size, point_dim, seq_len)
    return [left, right, points]

def get_init_inputs():
    return [channels, point_dim, mid_channels, out_dim]
