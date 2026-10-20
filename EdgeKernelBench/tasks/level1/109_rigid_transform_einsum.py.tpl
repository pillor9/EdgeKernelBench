# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: rigid_transform_einsum.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_109_rigid_transform_einsum',
 'level': 1,
 'name': 'rigid_transform_einsum',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['matrix_multiplication'],
 'semantic_references': [{'url': 'https://github.com/kornia/kornia', 'symbol': 'einsum_geometry'}],
 'summary': "rigid transform core; torch.einsum('bij,bnj->bni', R, pts)"}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, rotation: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bij,bnj->bni', rotation, points)
batch_size = __EKB_PARAM_batch_size__
num_points = __EKB_PARAM_num_points__
point_dim = __EKB_PARAM_point_dim__

def get_inputs():
    rotation = torch.rand(batch_size, point_dim, point_dim)
    points = torch.rand(batch_size, num_points, point_dim)
    return [rotation, points]

def get_init_inputs():
    return []
