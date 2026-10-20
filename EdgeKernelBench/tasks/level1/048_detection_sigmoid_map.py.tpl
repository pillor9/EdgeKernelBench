# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: detection_sigmoid_map.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_048_detection_sigmoid_map',
 'level': 1,
 'name': 'detection_sigmoid_map',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['activation'],
 'semantic_references': [{'url': 'https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py',
                          'symbol': 'Detect sigmoid logits'}],
 'summary': 'Apply sigmoid to raw detection score maps.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return []
