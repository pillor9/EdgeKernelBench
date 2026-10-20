# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: logsoftmax_axis_0.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_086_logsoftmax_axis_0',
 'level': 1,
 'name': 'logsoftmax_axis_0',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['activation'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Activation/LogSoftmax_axis_0.py',
                          'symbol': 'LogSoftmax_axis_0'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):

    def __init__(self, axis: int=0):
        super(Model, self).__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x, dim=self.axis)
dim0 = __EKB_PARAM_dim0__
dim1 = __EKB_PARAM_dim1__
dim2 = __EKB_PARAM_dim2__

def get_inputs():
    x = torch.randn(dim0, dim1, dim2)
    return [x]

def get_init_inputs():
    return [__EKB_INPUT_get_init_inputs_001_c001__]
