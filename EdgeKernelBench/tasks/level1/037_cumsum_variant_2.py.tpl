# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: cumsum_variant_2.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_037_cumsum_variant_2',
 'level': 1,
 'name': 'cumsum_variant_2',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['reduction'],
 'semantic_references': [{'url': 'https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/tools.py',
                          'symbol': 'cumsum_trick'}],
 'summary': 'cumsum_trick begins with x = x.cumsum(0); this Level1 task isolates that static cumsum'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(x, dim=0)
num_points = __EKB_PARAM_num_points__
feature_dim = __EKB_PARAM_feature_dim__

def get_inputs():
    return [torch.rand(num_points, feature_dim)]

def get_init_inputs():
    return []
