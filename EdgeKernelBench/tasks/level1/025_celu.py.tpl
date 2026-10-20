# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: celu.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_025_celu',
 'level': 1,
 'name': 'celu',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['activation'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Activation/Celu.py',
                          'symbol': 'Celu'}],
 'summary': 'torch.nn.functional.celu(x, alpha=self.alpha)'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.celu(x, alpha=self.alpha)
batch_size = __EKB_PARAM_batch_size__
dim = __EKB_PARAM_dim__
alpha = __EKB_PARAM_alpha__

def get_inputs():
    return [torch.rand(batch_size, dim)]

def get_init_inputs():
    return [alpha]
