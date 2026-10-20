# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: mobilenetv3_small.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_022_mobilenetv3_small',
 'level': 3,
 'name': 'mobilenetv3_small',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py'}],
 'summary': 'Source: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py '
            '(MobileNetV3-Small).',
 'adaptation_notes': ['Compact MobileNetV3-inspired classifier with an h-swish stem, four depthwise '
                      'inverted residual blocks, squeeze-excitation, projection, global pooling, and a '
                      'linear head.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class SE(nn.Module):

    def __init__(self, c: int):
        super().__init__()
        r = max(8, c // 4)
        self.a = nn.Conv2d(c, r, 1)
        self.b = nn.Conv2d(r, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.hardsigmoid(self.b(F.relu(self.a(F.adaptive_avg_pool2d(x, 1)))))

class IR(nn.Module):

    def __init__(self, ci: int, ce: int, co: int, s: int, se: bool):
        super().__init__()
        self.use = ci == co and s == 1
        self.e = nn.Conv2d(ci, ce, 1, bias=False)
        self.eb = nn.BatchNorm2d(ce)
        self.d = nn.Conv2d(ce, ce, 3, stride=s, padding=1, groups=ce, bias=False)
        self.db = nn.BatchNorm2d(ce)
        self.se = SE(ce) if se else nn.Identity()
        self.p = nn.Conv2d(ce, co, 1, bias=False)
        self.pb = nn.BatchNorm2d(co)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.hardswish(self.eb(self.e(x)))
        y = F.hardswish(self.db(self.d(y)))
        y = self.pb(self.p(self.se(y)))
        return x + y if self.use else y

class Model(nn.Module):

    def __init__(self, c1: int, c2: int, c3: int, classes: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, c1, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c1), nn.Hardswish())
        self.body = nn.Sequential(IR(c1, 2 * c1, c1, 1, True), IR(c1, 3 * c1, c2, 2, False), IR(c2, 3 * c2, c2, 1, True), IR(c2, 3 * c2, c3, 2, True))
        self.last = nn.Conv2d(c3, 4 * c3, 1)
        self.head = nn.Linear(4 * c3, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.hardswish(self.last(self.body(self.stem(x))))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))
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
