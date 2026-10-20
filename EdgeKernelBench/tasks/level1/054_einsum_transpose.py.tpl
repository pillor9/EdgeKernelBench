# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: einsum_transpose.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_054_einsum_transpose',
 'level': 1,
 'name': 'einsum_transpose',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['matrix_multiplication', 'tensor_transform'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Matrix/Einsum_transpose.py',
                          'symbol': 'Einsum_transpose'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):

    def __init__(self, equation: str='ij->ji'):
        super(Model, self).__init__()
        self.equation = equation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum(self.equation, x)
M = __EKB_PARAM_M__
N = __EKB_PARAM_N__
equation = __EKB_PARAM_equation__

def get_inputs():
    x = torch.randn(M, N)
    return [x]

def get_init_inputs():
    return [equation]
