# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: reshape.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_106_reshape',
 'level': 1,
 'name': 'reshape',
 'scenario_tags': ['mobile_personal', 'embodied_robotics'],
 'operator_tags': ['tensor_transform'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Tensor/Reshape.py',
                          'symbol': 'Reshape'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, target_shape):
        super(Model, self).__init__()
        self.target_shape = tuple(target_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.reshape(x, self.target_shape)
batch_size = __EKB_PARAM_batch_size__
input_shape = __EKB_PARAM_input_shape__
target_shape = __EKB_PARAM_target_shape__

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [target_shape]
