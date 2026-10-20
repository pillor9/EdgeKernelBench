# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: globalaveragepool_3d.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_071_globalaveragepool_3d',
 'level': 1,
 'name': 'globalaveragepool_3d',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['pooling'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Pooling/GlobalAveragePool_3d.py',
                          'symbol': 'GlobalAveragePool_3d'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool3d(x, output_size=1)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
depth = __EKB_PARAM_depth__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    return []
