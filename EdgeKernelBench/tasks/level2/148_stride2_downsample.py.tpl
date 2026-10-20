# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: stride2_downsample.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_148_stride2_downsample',
 'level': 2,
 'name': 'stride2_downsample',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'normalization'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py',
                          'symbol': 'downsample'}],
 'summary': 'ResNet downsample branch: conv1x1(stride=2) -> BatchNorm2d'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels]
