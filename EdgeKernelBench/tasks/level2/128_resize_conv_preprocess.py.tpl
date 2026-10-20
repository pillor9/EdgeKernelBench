# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: resize_conv_preprocess.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_128_resize_conv_preprocess',
 'level': 2,
 'name': 'resize_conv_preprocess',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution', 'matrix_multiplication', 'resampling'],
 'semantic_references': [{'url': 'https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py',
                          'symbol': 'interpolate_conv'}],
 'summary': 'static multi-stream NVR-style bilinear resize followed by a Conv2d preprocessing projection'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(resized_height, resized_width), mode='bilinear', align_corners=False)
        return self.conv(x)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
resized_height = __EKB_PARAM_resized_height__
resized_width = __EKB_PARAM_resized_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, out_channels]
