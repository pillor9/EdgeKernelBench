# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: squeeze_excitation.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_143_squeeze_excitation',
 'level': 2,
 'name': 'squeeze_excitation',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'activation', 'pooling', 'reduction', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py',
                          'symbol': 'SqueezeExcitation'}],
 'summary': 'torchvision SqueezeExcitation: global avgpool -> 1x1 conv -> ReLU -> 1x1 conv -> Hardsigmoid -> '
            'scale input'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, squeeze_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, squeeze_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Conv2d(squeeze_channels, channels, kernel_size=1)
        self.scale_activation = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=(-2, -1), keepdim=True)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return scale * x
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
squeeze_channels = __EKB_PARAM_squeeze_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, squeeze_channels]
