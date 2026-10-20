# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: shufflenetv2_05.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_033_shufflenetv2_05',
 'level': 3,
 'name': 'shufflenetv2_05',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'tensor_transform'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py'}],
 'summary': 'Source: https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py '
            '(ShuffleNetV2 0.5x).',
 'adaptation_notes': ['Compact ShuffleNetV2-inspired classifier with four shuffle units, stride-2 dual '
                      'branches, stride-1 split branches, global pooling, and a linear head.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

def shuffle(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    return x.reshape(b, 2, c // 2, h, w).transpose(1, 2).reshape(b, c, h, w)

class Unit(nn.Module):

    def __init__(self, ci: int, co: int, stride: int):
        super().__init__()
        half = co // 2
        self.s = stride
        self.b1 = nn.Sequential(nn.Conv2d(ci, ci, 3, stride=2, padding=1, groups=ci, bias=False), nn.BatchNorm2d(ci), nn.Conv2d(ci, half, 1, bias=False), nn.BatchNorm2d(half), nn.ReLU()) if stride == 2 else nn.Identity()
        src = ci if stride == 2 else ci // 2
        self.b2 = nn.Sequential(nn.Conv2d(src, half, 1, bias=False), nn.BatchNorm2d(half), nn.ReLU(), nn.Conv2d(half, half, 3, stride=stride, padding=1, groups=half, bias=False), nn.BatchNorm2d(half), nn.Conv2d(half, half, 1, bias=False), nn.BatchNorm2d(half), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.s == 1:
            x1, x2 = torch.chunk(x, 2, dim=1)
            y = torch.cat([x1, self.b2(x2)], 1)
        else:
            y = torch.cat([self.b1(x), self.b2(x)], 1)
        return shuffle(y)

class Model(nn.Module):

    def __init__(self, c1: int, c2: int, c3: int, classes: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, c1, 3, stride=2, padding=1), nn.BatchNorm2d(c1), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1))
        self.body = nn.Sequential(Unit(c1, c2, 2), Unit(c2, c2, 1), Unit(c2, c3, 2), Unit(c3, c3, 1))
        self.head = nn.Linear(c3, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(F.adaptive_avg_pool2d(self.body(self.stem(x)), 1).flatten(1))
batch_size = __EKB_PARAM_batch_size__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
c1 = __EKB_PARAM_c1__
c2 = __EKB_PARAM_c2__
c3 = __EKB_PARAM_c3__
classes = __EKB_PARAM_classes__

def get_inputs():
    return [torch.randn(batch_size, __EKB_INPUT_get_inputs_001_c001__, height, width)]

def get_init_inputs():
    return [c1, c2, c3, classes]
