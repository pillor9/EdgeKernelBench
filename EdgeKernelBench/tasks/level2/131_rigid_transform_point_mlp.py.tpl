# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: rigid_transform_point_mlp.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_131_rigid_transform_point_mlp',
 'level': 2,
 'name': 'rigid_transform_point_mlp',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['matrix_multiplication', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/kornia/kornia/blob/main/kornia/geometry/linalg.py',
                          'symbol': 'transform_points'}],
 'summary': 'Kornia homogeneous point transform followed by a per-point MLP'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, num_points, hidden_dim, out_dim):
        super().__init__()
        self.register_buffer('ones', torch.ones(1, num_points, 1))
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        ones = self.ones.expand(points.shape[0], -1, -1)
        points_h = torch.cat([points, ones], dim=2)
        transformed_h = torch.bmm(points_h, transform.transpose(1, 2))
        transformed = transformed_h[:, :, :3] / transformed_h[:, :, 3:4]
        return self.fc2(F.relu(self.fc1(transformed)))
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
hidden_dim = __EKB_PARAM_hidden_dim__
out_dim = __EKB_PARAM_out_dim__

def get_inputs():
    points = torch.rand(batch_size, seq_len, __EKB_INPUT_get_inputs_001_c001__)
    transform = torch.eye(__EKB_INPUT_get_inputs_001_c002__).unsqueeze(__EKB_INPUT_get_inputs_001_c003__).repeat(batch_size, __EKB_INPUT_get_inputs_001_c004__, __EKB_INPUT_get_inputs_001_c005__)
    transform[:, :__EKB_INPUT_get_inputs_001_c006__, __EKB_INPUT_get_inputs_001_c007__] = torch.rand(batch_size, __EKB_INPUT_get_inputs_001_c008__)
    return [points, transform]

def get_init_inputs():
    return [seq_len, hidden_dim, out_dim]
