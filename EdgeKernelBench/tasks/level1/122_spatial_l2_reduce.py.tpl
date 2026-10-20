# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: spatial_l2_reduce.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_122_spatial_l2_reduce',
 'level': 1,
 'name': 'spatial_l2_reduce',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['reduction', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/openvinotoolkit/anomalib', 'symbol': 'anomaly_l2'}],
 'summary': 'Square, sum over spatial dimensions, and take the square root.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.sum(torch.pow(x, 2), dim=(2, 3)))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return []
