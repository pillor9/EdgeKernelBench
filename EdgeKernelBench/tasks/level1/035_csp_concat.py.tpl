# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: csp_concat.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_035_csp_concat',
 'level': 1,
 'name': 'csp_concat',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['tensor_transform'],
 'semantic_references': [{'url': 'https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py',
                          'symbol': 'C2f torch.cat'}],
 'summary': 'Concatenate two feature branches along the channel axis, following the C2f concatenation '
            'pattern.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.cat([x1, x2], dim=1)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    x1 = torch.rand(batch_size, channels, height, width)
    x2 = torch.rand(batch_size, channels, height, width)
    return [x1, x2]

def get_init_inputs():
    return []
