# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: onehot.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_096_onehot',
 'level': 1,
 'name': 'onehot',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['tensor_transform', 'tensor_generation', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Tensor/OneHot.py',
                          'symbol': 'OneHot'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn.functional as F
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, depth, on_value=1, off_value=0, axis=-1):
        super(Model, self).__init__()
        self.depth = depth
        self.on_value = on_value
        self.off_value = off_value
        self.axis = axis

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        one_hot_result = F.one_hot(indices, num_classes=self.depth)
        one_hot_result = one_hot_result * (self.on_value - self.off_value) + self.off_value
        if self.axis != -1:
            one_hot_result = one_hot_result.movedim(-1, self.axis)
        return one_hot_result
batch_size = __EKB_PARAM_batch_size__
dim1 = __EKB_PARAM_dim1__
dim2 = __EKB_PARAM_dim2__
depth = __EKB_PARAM_depth__

def get_inputs() -> list:
    x = torch.randint(__EKB_INPUT_get_inputs_001_c001__, depth, (batch_size, dim1))
    return [x]

def get_init_inputs():
    return [depth]
