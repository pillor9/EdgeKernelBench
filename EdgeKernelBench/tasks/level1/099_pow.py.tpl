# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: pow.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_099_pow',
 'level': 1,
 'name': 'pow',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['elementwise'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Binary/Pow.py',
                          'symbol': 'Pow'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.pow(x, y)
dim1 = __EKB_PARAM_dim1__
dim2 = __EKB_PARAM_dim2__
dim3 = __EKB_PARAM_dim3__

def get_inputs():
    torch.manual_seed(__EKB_INPUT_get_inputs_001_c001__)
    x = torch.rand(dim1, dim2, dim3) * __EKB_INPUT_get_inputs_001_c002__ + __EKB_INPUT_get_inputs_001_c003__
    y = torch.rand(dim1, dim2, dim3) * __EKB_INPUT_get_inputs_001_c004__ + __EKB_INPUT_get_inputs_001_c005__
    return [x, y]

def get_init_inputs():
    return []
