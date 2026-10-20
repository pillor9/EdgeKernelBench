# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: gather.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_058_gather',
 'level': 1,
 'name': 'gather',
 'scenario_tags': ['mobile_personal', 'embodied_robotics'],
 'operator_tags': ['tensor_transform', 'type_conversion', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Tensor/Gather.py',
                          'symbol': 'Gather'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, axis: int):
        super(Model, self).__init__()
        self.axis = axis

    def forward(self, params: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        assert indices.dim() == 1, 'This standard case expects 1D indices'
        assert indices.dtype in (torch.int32, torch.int64)
        if indices.dtype != torch.int64:
            indices = indices.to(torch.int64)
        axis = self.axis if self.axis >= 0 else self.axis + params.dim()
        return torch.index_select(params, dim=axis, index=indices)
B, C, D = __EKB_PARAM_B_C_D__
axis = __EKB_PARAM_axis__

def get_inputs():
    params = torch.rand(B, C, D)
    idx = torch.randint(low=__EKB_INPUT_get_inputs_001_c001__, high=C, size=(__EKB_INPUT_get_inputs_001_c002__,), dtype=torch.int64)
    return [params, idx]

def get_init_inputs():
    return [axis]
