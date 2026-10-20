# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: gemm_transposed.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_066_gemm_transposed',
 'level': 1,
 'name': 'gemm_transposed',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['matrix_multiplication', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Matrix/Gemm_transposed.py',
                          'symbol': 'Gemm_transposed'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):

    def __init__(self, in_features: int, out_features: int, alpha: float=1.0):
        super(Model, self).__init__()
        self.alpha = alpha
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.t()
        return self.alpha * self.linear(x_t)
batch_size = __EKB_PARAM_batch_size__
in_features = __EKB_PARAM_in_features__
out_features = __EKB_PARAM_out_features__
alpha = __EKB_PARAM_alpha__

def get_inputs():
    x = torch.randn(in_features, batch_size)
    return [x]

def get_init_inputs():
    return [in_features, out_features, alpha]
