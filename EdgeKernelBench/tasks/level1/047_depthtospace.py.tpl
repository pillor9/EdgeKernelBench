# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: depthtospace.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_047_depthtospace',
 'level': 1,
 'name': 'depthtospace',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['tensor_transform'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Tensor/DepthToSpace.py',
                          'symbol': 'DepthToSpace'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, block_size: int):
        super(Model, self).__init__()
        self.block_size = block_size
        self.op = nn.PixelShuffle(block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)
batch_size = __EKB_PARAM_batch_size__
block_size = __EKB_PARAM_block_size__
out_channels = __EKB_PARAM_out_channels__
out_height = __EKB_PARAM_out_height__
out_width = __EKB_PARAM_out_width__
in_channels = __EKB_PARAM_in_channels__
in_height = __EKB_PARAM_in_height__
in_width = __EKB_PARAM_in_width__
input_shape = __EKB_PARAM_input_shape__

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [block_size]
