# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: batchnormalization_epsilon.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_020_batchnormalization_epsilon',
 'level': 1,
 'name': 'batchnormalization_epsilon',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['normalization'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Normalization/BatchNormalization_epsilon.py',
                          'symbol': 'BatchNormalization_epsilon'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, num_features: int, eps: float=1e-05):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)
batch_size = __EKB_PARAM_batch_size__
features = __EKB_PARAM_features__
dim1 = __EKB_PARAM_dim1__
dim2 = __EKB_PARAM_dim2__
epsilon = __EKB_PARAM_epsilon__

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features, epsilon]
