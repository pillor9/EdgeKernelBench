# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: globalmaxpool.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_072_globalmaxpool',
 'level': 1,
 'name': 'globalmaxpool',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['pooling', 'reduction', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Pooling/GlobalMaxPool.py',
                          'symbol': 'GlobalMaxPool'}],
 'summary': 'source global max pool over all spatial dimensions'}
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.max(x, dim=-1, keepdim=True).values
        y = torch.max(y, dim=-2, keepdim=True).values
        return y
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    torch.manual_seed(__EKB_INPUT_get_inputs_001_c001__)
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return []
