# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: se_squeeze_path.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_117_se_squeeze_path',
 'level': 1,
 'name': 'se_squeeze_path',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution', 'activation', 'pooling'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py',
                          'symbol': 'SqueezeExcitation'}],
 'summary': 'MobileNetV3 uses torchvision.ops.misc.SqueezeExcitation with Hardsigmoid scale activation'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, input_channels: int, squeeze_channels: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.activation = nn.ReLU()
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.scale_activation = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)
batch_size = __EKB_PARAM_batch_size__
input_channels = __EKB_PARAM_input_channels__
squeeze_channels = __EKB_PARAM_squeeze_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [input_channels, squeeze_channels]
