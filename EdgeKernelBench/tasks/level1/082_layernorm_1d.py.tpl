# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: layernorm_1d.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_082_layernorm_1d',
 'level': 1,
 'name': 'layernorm_1d',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['normalization'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Normalization/LayerNorm_1d.py',
                          'symbol': 'LayerNorm_1d'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn
from typing import List, Tuple

class Model(nn.Module):

    def __init__(self, normalized_shape: int, eps: float=1e-05, elementwise_affine: bool=True):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
features = __EKB_PARAM_features__
eps = __EKB_PARAM_eps__
elementwise_affine = __EKB_PARAM_elementwise_affine__

def get_inputs():
    x = torch.randn(batch_size, seq_len, features)
    return [x]

def get_init_inputs():
    return [features, eps, elementwise_affine]
