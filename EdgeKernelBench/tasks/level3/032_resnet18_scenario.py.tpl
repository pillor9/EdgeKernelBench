# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: resnet18_scenario.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_032_resnet18_scenario',
 'level': 3,
 'name': 'resnet18_scenario',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py'}],
 'summary': 'Source: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py '
            '(resnet18/BasicBlock).',
 'adaptation_notes': ['complete ResNet-18 stem, BasicBlock stage depths [2,2,2,2], global spatial mean, and '
                      'classifier; dimensions are supplied by the external profile.']}
import torch
import torch.nn as nn

def conv3(ci: int, co: int, stride: int=1):
    return nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):

    def __init__(self, ci: int, co: int, stride: int=1):
        super().__init__()
        self.conv1 = conv3(ci, co, stride)
        self.bn1 = nn.BatchNorm2d(co)
        self.relu = nn.ReLU()
        self.conv2 = conv3(co, co)
        self.bn2 = nn.BatchNorm2d(co)
        self.short = nn.Sequential(nn.Conv2d(ci, co, 1, stride=stride, bias=False), nn.BatchNorm2d(co)) if stride != 1 or ci != co else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + self.short(x))

class Model(nn.Module):

    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(channels, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1))
        self.layers = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64), BasicBlock(64, 128, 2), BasicBlock(128, 128), BasicBlock(128, 256, 2), BasicBlock(256, 256), BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.fc = nn.Linear(512, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.layers(self.stem(x)).mean(dim=(-2, -1)))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
classes = __EKB_PARAM_classes__

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, classes]
