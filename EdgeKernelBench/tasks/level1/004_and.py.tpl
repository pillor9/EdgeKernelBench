# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: and.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_004_and',
 'level': 1,
 'name': 'and',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['elementwise'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Logic/And.py',
                          'symbol': 'And'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_bool = a.bool()
        b_bool = b.bool()
        return torch.logical_and(a_bool, b_bool)
batch_size = __EKB_PARAM_batch_size__
H, W = __EKB_PARAM_H_W__

def get_inputs():
    a = torch.rand(batch_size, H, W) > __EKB_INPUT_get_inputs_001_c001__
    b = torch.rand(batch_size, H, W) > __EKB_INPUT_get_inputs_001_c002__
    return [a, b]

def get_init_inputs():
    return []
