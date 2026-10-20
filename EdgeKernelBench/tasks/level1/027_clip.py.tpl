# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: clip.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_027_clip',
 'level': 1,
 'name': 'clip',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['elementwise'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Tensor/Clip.py',
                          'symbol': 'Clip'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, min_val: float=0.0, max_val: float=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=self.min_val, max=self.max_val)
batch_size = __EKB_PARAM_batch_size__
input_shape = __EKB_PARAM_input_shape__
min_val = __EKB_PARAM_min_val__
max_val = __EKB_PARAM_max_val__

def get_inputs() -> list:
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [min_val, max_val]
