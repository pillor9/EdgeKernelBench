# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: add.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_002_add',
 'level': 1,
 'name': 'add',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['elementwise'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Binary/Add.py',
                          'symbol': 'Add'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.add(a, b)
batch_size = __EKB_PARAM_batch_size__
C, H, W = __EKB_PARAM_C_H_W__

def get_inputs():
    a = torch.rand(batch_size, C, H, W)
    b = torch.rand(__EKB_INPUT_get_inputs_001_c001__, C, __EKB_INPUT_get_inputs_001_c002__, __EKB_INPUT_get_inputs_001_c003__)
    return [a, b]

def get_init_inputs():
    return []
