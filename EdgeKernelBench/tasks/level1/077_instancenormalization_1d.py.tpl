# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: instancenormalization_1d.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_077_instancenormalization_1d',
 'level': 1,
 'name': 'instancenormalization_1d',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['normalization'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Normalization/InstanceNormalization_1d.py',
                          'symbol': 'InstanceNormalization_1d'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):

    def __init__(self, num_features: int, eps: float=1e-05, affine: bool=False):
        super(Model, self).__init__()
        self.inorm = nn.InstanceNorm1d(num_features=num_features, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inorm(x)
batch_size = __EKB_PARAM_batch_size__
num_features = __EKB_PARAM_num_features__
length = __EKB_PARAM_length__
eps = __EKB_PARAM_eps__
affine = __EKB_PARAM_affine__

def get_inputs():
    x = torch.randn(batch_size, num_features, length)
    return [x]

def get_init_inputs():
    return [num_features, eps, affine]
