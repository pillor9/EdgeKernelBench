# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: einsum_diagonal.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_051_einsum_diagonal',
 'level': 1,
 'name': 'einsum_diagonal',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['reduction', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Matrix/Einsum_diagonal.py',
                          'symbol': 'Einsum_diagonal'}],
 'summary': 'source torch.einsum("ii->i", x), extracting the diagonal of a square matrix'}
import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):

    def __init__(self, size: int):
        super(Model, self).__init__()
        self.register_buffer('eye', torch.eye(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.eye).sum(dim=1)
N = __EKB_PARAM_N__

def get_inputs():
    x = torch.randn(N, N)
    return [x]

def get_init_inputs():
    return [N]
