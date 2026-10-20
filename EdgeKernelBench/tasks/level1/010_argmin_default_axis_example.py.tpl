# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: argmin_default_axis_example.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_010_argmin_default_axis_example',
 'level': 1,
 'name': 'argmin_default_axis_example',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['reduction'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Reduction/ArgMin_default_axis_example.py',
                          'symbol': 'ArgMin_default_axis_example'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, data) -> torch.Tensor:
        return torch.argmin(data)
batch_size = __EKB_PARAM_batch_size__
dim1 = __EKB_PARAM_dim1__
dim2 = __EKB_PARAM_dim2__

def get_inputs() -> List[torch.Tensor]:
    return [torch.randn(batch_size, dim1, dim2)]

def get_init_inputs() -> List:
    return []
