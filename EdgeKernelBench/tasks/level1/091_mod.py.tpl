# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: mod.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_091_mod',
 'level': 1,
 'name': 'mod',
 'scenario_tags': ['mobile_personal', 'smart_aiot'],
 'operator_tags': ['elementwise'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Binary/Mod.py',
                          'symbol': 'Mod'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):

    def __init__(self, fmod: int=0):
        super(Model, self).__init__()
        self.fmod = fmod

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.fmod == 1:
            return torch.fmod(a, b)
        else:
            return torch.remainder(a, b)
dim1 = __EKB_PARAM_dim1__
dim2 = __EKB_PARAM_dim2__
dim3 = __EKB_PARAM_dim3__
fmod = __EKB_PARAM_fmod__

def get_inputs():
    a = __EKB_INPUT_get_inputs_001_c001__ * torch.rand(dim1, dim2, dim3)
    b = __EKB_INPUT_get_inputs_001_c002__ + __EKB_INPUT_get_inputs_001_c003__ * torch.rand(dim1, dim2, dim3)
    return [a, b]

def get_init_inputs():
    return [fmod]
