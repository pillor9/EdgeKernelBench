# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: gemm_beta.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_063_gemm_beta',
 'level': 1,
 'name': 'gemm_beta',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['matrix_multiplication', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Matrix/Gemm_beta.py',
                          'symbol': 'Gemm_beta'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):

    def __init__(self, alpha: float=0.5, beta: float=1.0, transA: int=0, transB: int=0):
        super(Model, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.transA = transA
        self.transB = transB

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        a_mat = a.T if self.transA == 1 else a
        b_mat = b.T if self.transB == 1 else b
        return self.alpha * torch.mm(a_mat, b_mat) + self.beta * c
M = __EKB_PARAM_M__
K = __EKB_PARAM_K__
N = __EKB_PARAM_N__
alpha = __EKB_PARAM_alpha__
beta = __EKB_PARAM_beta__
transA = __EKB_PARAM_transA__
transB = __EKB_PARAM_transB__

def get_inputs():
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    c = torch.randn(__EKB_INPUT_get_inputs_001_c001__, N)
    return [a, b, c]

def get_init_inputs():
    return [alpha, beta, transA, transB]
