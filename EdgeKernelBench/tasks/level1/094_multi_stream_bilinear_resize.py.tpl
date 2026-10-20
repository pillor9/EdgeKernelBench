# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: multi_stream_bilinear_resize.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_094_multi_stream_bilinear_resize',
 'level': 1,
 'name': 'multi_stream_bilinear_resize',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['resampling'],
 'semantic_references': [],
 'summary': 'Bilinear resize of image streams to profile-specified output dimensions.'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, out_size: int):
        super().__init__()
        self.out_size = out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
out_size = __EKB_PARAM_out_size__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [out_size]
