# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: fc_autoencoder.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_081_fc_autoencoder',
 'level': 2,
 'name': 'fc_autoencoder',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['matrix_multiplication', 'normalization', 'activation', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/keras_model.py',
                          'symbol': 'get_model'}],
 'summary': 'MLPerf Tiny anomaly model dense autoencoder pattern 128-128-128-128-8-128-128-128-128-output '
            'with BN and ReLU between Dense layers'}
import torch
import torch.nn as nn

class DenseBNRelu(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.fc(x)))

class Model(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(DenseBNRelu(dim, 128), DenseBNRelu(128, 128), DenseBNRelu(128, 128), DenseBNRelu(128, 128), DenseBNRelu(128, 8), DenseBNRelu(8, 128), DenseBNRelu(128, 128), DenseBNRelu(128, 128), DenseBNRelu(128, 128), nn.Linear(128, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
batch_size = __EKB_PARAM_batch_size__
dim = __EKB_PARAM_dim__

def get_inputs():
    return [torch.rand(batch_size, dim)]

def get_init_inputs():
    return [dim]
