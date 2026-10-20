# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: cumsum.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_036_cumsum',
 'level': 1,
 'name': 'cumsum',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['reduction'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Tensor/CumSum.py',
                          'symbol': 'CumSum'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, axis: int=-1):
        super(Model, self).__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(x, dim=self.axis)
batch_size = __EKB_PARAM_batch_size__
input_shape = __EKB_PARAM_input_shape__

def get_inputs():
    x = __EKB_INPUT_get_inputs_001_c001__ * torch.rand(batch_size, *input_shape) - __EKB_INPUT_get_inputs_001_c002__
    return [x]

def get_init_inputs():
    return []
