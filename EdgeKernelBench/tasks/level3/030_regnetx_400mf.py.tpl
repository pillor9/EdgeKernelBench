# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: regnetx_400mf.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_030_regnetx_400mf',
 'level': 3,
 'name': 'regnetx_400mf',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py'}],
 'summary': 'Source: https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py '
            '(RegNetX-400MF).',
 'adaptation_notes': ['compact static RegNetX-style stem, two grouped bottleneck stages with projections, '
                      'global average, and classifier.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, groups: int, stride: int):
        super().__init__()
        width = out_channels
        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.proj = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels)) if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        return F.relu(self.bn3(self.conv3(y)) + self.proj(x))

class Model(nn.Module):

    def __init__(self, stem_channels: int, stage1_channels: int, stage2_channels: int, groups: int, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, stem_channels, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(stem_channels), nn.ReLU())
        self.stage1 = nn.Sequential(Bottleneck(stem_channels, stage1_channels, groups, 2), Bottleneck(stage1_channels, stage1_channels, groups, 1))
        self.stage2 = nn.Sequential(Bottleneck(stage1_channels, stage2_channels, groups, 2), Bottleneck(stage2_channels, stage2_channels, groups, 1))
        self.head = nn.Linear(stage2_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stage2(self.stage1(self.stem(x))).mean(dim=(2, 3)))
batch_size = __EKB_PARAM_batch_size__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
stem_channels = __EKB_PARAM_stem_channels__
stage1_channels = __EKB_PARAM_stage1_channels__
stage2_channels = __EKB_PARAM_stage2_channels__
groups = __EKB_PARAM_groups__
num_classes = __EKB_PARAM_num_classes__

def get_inputs():
    return [torch.randn(batch_size, __EKB_INPUT_get_inputs_001_c001__, height, width)]

def get_init_inputs():
    return [stem_channels, stage1_channels, stage2_channels, groups, num_classes]
