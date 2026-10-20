# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/fxia22--pointnet.pytorch/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: stn3d_static.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_147_stn3d_static',
 'level': 2,
 'name': 'stn3d_static',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py',
                          'symbol': 'STN3d'}],
 'summary': 'PointNet STN3d: Conv1d/BN/ReLU stack -> max over points -> FC/BN/ReLU stack -> 3x3 transform'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.register_buffer('identity', torch.eye(3).reshape(1, 9))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        x = x + self.identity
        return x.view(batch_size, 3, 3)
batch_size = __EKB_PARAM_batch_size__
num_points = __EKB_PARAM_num_points__

def get_inputs():
    return [torch.rand(batch_size, __EKB_INPUT_get_inputs_001_c001__, num_points)]

def get_init_inputs():
    return []
