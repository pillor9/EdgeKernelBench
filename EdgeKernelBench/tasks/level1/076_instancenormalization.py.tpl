# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: instancenormalization.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_076_instancenormalization',
 'level': 1,
 'name': 'instancenormalization',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['normalization'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Normalization/InstanceNormalization.py',
                          'symbol': 'InstanceNormalization'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, num_features: int):
        super(Model, self).__init__()
        self.inorm = nn.InstanceNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inorm(x)
batch_size = __EKB_PARAM_batch_size__
features = __EKB_PARAM_features__
dim1 = __EKB_PARAM_dim1__
dim2 = __EKB_PARAM_dim2__

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]
