# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: atan.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_013_atan',
 'level': 1,
 'name': 'atan',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['elementwise'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Trigonometry/ATan.py',
                          'symbol': 'ATan'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.atan(x)
batch_size = __EKB_PARAM_batch_size__
input_shape = __EKB_PARAM_input_shape__

def get_inputs():
    x = __EKB_INPUT_get_inputs_001_c001__ * torch.rand(batch_size, *input_shape) - __EKB_INPUT_get_inputs_001_c002__
    return [x]

def get_init_inputs():
    return []
