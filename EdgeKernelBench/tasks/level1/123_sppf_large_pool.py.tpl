# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: sppf_large_pool.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_123_sppf_large_pool',
 'level': 1,
 'name': 'sppf_large_pool',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['pooling'],
 'semantic_references': [{'url': 'https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py',
                          'symbol': 'SPPF.m'}],
 'summary': 'SPPF max-pooling primitive with stride 1 and same-size padding; the kernel size is supplied by '
            'the profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, kernel_size: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [kernel_size]
