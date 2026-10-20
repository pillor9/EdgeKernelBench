# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: mobilenetv2.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_020_mobilenetv2',
 'level': 3,
 'name': 'mobilenetv2',
 'scenario_tags': ['mobile_personal', 'smart_aiot'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'reduction',
                   'residual',
                   'type_conversion',
                   'tensor_generation',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py',
                          'symbol': 'MobileNetV2 / InvertedResidual'}],
 'summary': 'torchvision MobileNetV2 feature stack with inverted residual settings [1,16,1,1] through '
            '[6,320,1,1] and final classifier'}
import torch
import torch.nn as nn

class ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6())

class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))
        layers.extend([ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim), nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(out_channels)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class Model(nn.Module):

    def __init__(self, input_channels, num_classes):
        super().__init__()
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        features = [ConvBNReLU(input_channels, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Linear(last_channel, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=(-2, -1))
        return self.classifier(x)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
num_classes = __EKB_PARAM_num_classes__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, num_classes]
