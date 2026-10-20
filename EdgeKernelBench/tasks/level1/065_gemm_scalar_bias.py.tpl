# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: gemm_scalar_bias.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_065_gemm_scalar_bias',
 'level': 1,
 'name': 'gemm_scalar_bias',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['matrix_multiplication', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Matrix/Gemm_scalar_bias.py',
                          'symbol': 'Gemm_scalar_bias'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_features: int, out_features: int, alpha: float=1.0, beta: float=1.0, with_bias: bool=True):
        super(Model, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.with_bias = with_bias
        self.linear = nn.Linear(in_features, out_features, bias=with_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.alpha * self.linear(x)
        return y
batch_size = __EKB_PARAM_batch_size__
in_features = __EKB_PARAM_in_features__
out_features = __EKB_PARAM_out_features__
alpha = __EKB_PARAM_alpha__
beta = __EKB_PARAM_beta__
with_bias = __EKB_PARAM_with_bias__

def get_inputs():
    x = torch.rand(batch_size, in_features)
    return [x]

def get_init_inputs():
    return [in_features, out_features, alpha, beta, with_bias]
