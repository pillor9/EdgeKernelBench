# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: gatherelements.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_059_gatherelements',
 'level': 1,
 'name': 'gatherelements',
 'scenario_tags': ['mobile_personal', 'embodied_robotics'],
 'operator_tags': ['tensor_transform', 'type_conversion', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Tensor/GatherElements.py',
                          'symbol': 'GatherElements'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, axis: int):
        super(Model, self).__init__()
        self.axis = axis

    def forward(self, data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        assert data.shape == indices.shape, 'data and indices must have the same shape'
        if indices.dtype != torch.int64:
            indices = indices.to(torch.int64)
        axis = self.axis if self.axis >= 0 else self.axis + data.dim()
        return torch.gather(data, dim=axis, index=indices)
B, C, D = __EKB_PARAM_B_C_D__
axis = __EKB_PARAM_axis__

def get_inputs():
    data = torch.rand(B, C, D)
    idx = torch.randint(low=__EKB_INPUT_get_inputs_001_c001__, high=C, size=(B, C, D), dtype=torch.int64)
    return [data, idx]

def get_init_inputs():
    return [axis]
