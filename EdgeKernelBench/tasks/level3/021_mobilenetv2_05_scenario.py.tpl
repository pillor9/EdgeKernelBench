# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: mobilenetv2_05_scenario.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_021_mobilenetv2_05_scenario',
 'level': 3,
 'name': 'mobilenetv2_05_scenario',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'reduction',
                   'residual',
                   'type_conversion',
                   'tensor_generation',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py'}],
 'summary': 'Source: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py '
            '(MobileNetV2/InvertedResidual).',
 'adaptation_notes': ['complete width-0.5 MobileNetV2 inverted-residual setting [1,16,1,1] through '
                      '[6,320,1,1], final 1x1 feature layer, global mean, and classifier.']}
import math
import torch
import torch.nn as nn

def divisible(v: float, divisor: int=8) -> int:
    value = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return value + divisor if value < 0.9 * v else value

class ConvBNReLU(nn.Sequential):

    def __init__(self, ci: int, co: int, kernel: int=3, stride: int=1, groups: int=1):
        super().__init__(nn.Conv2d(ci, co, kernel, stride=stride, padding=(kernel - 1) // 2, groups=groups, bias=False), nn.BatchNorm2d(co), nn.ReLU6())

class InvertedResidual(nn.Module):

    def __init__(self, ci: int, co: int, stride: int, expand: int):
        super().__init__()
        hidden = int(round(ci * expand))
        self.residual = stride == 1 and ci == co
        layers = []
        if expand != 1:
            layers.append(ConvBNReLU(ci, hidden, kernel=1))
        layers.extend([ConvBNReLU(hidden, hidden, stride=stride, groups=hidden), nn.Conv2d(hidden, co, 1, bias=False), nn.BatchNorm2d(co)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.residual else self.conv(x)

class Model(nn.Module):

    def __init__(self, channels: int, width_mult: float, classes: int):
        super().__init__()
        ci = divisible(32 * width_mult)
        last = divisible(1280 * max(1.0, width_mult))
        features = [ConvBNReLU(channels, ci, stride=2)]
        for expand, out, repeats, first_stride in [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]:
            co = divisible(out * width_mult)
            for index in range(repeats):
                features.append(InvertedResidual(ci, co, first_stride if index == 0 else 1, expand))
                ci = co
        features.append(ConvBNReLU(ci, last, kernel=1))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Linear(last, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).mean(dim=(-2, -1)))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
width_mult = __EKB_PARAM_width_mult__
classes = __EKB_PARAM_classes__

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, width_mult, classes]
