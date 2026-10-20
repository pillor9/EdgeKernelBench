# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: elu.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_055_elu',
 'level': 1,
 'name': 'elu',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['activation'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Activation/ELU.py',
                          'symbol': 'ELU'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):

    def __init__(self, alpha: float=1.0):
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, alpha=self.alpha)
dim1 = __EKB_PARAM_dim1__
dim2 = __EKB_PARAM_dim2__
dim3 = __EKB_PARAM_dim3__
alpha = __EKB_PARAM_alpha__

def get_inputs():
    x = torch.randn(dim1, dim2, dim3)
    return [x]

def get_init_inputs():
    return [alpha]
