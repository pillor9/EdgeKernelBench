# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: resnet_stem.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_130_resnet_stem',
 'level': 2,
 'name': 'resnet_stem',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'pooling'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py'}],
 'summary': 'Source: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py (ResNet stem).',
 'adaptation_notes': ['conv7x7 stride2, BatchNorm, ReLU, and maxpool3x3 stride2.']}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels]
