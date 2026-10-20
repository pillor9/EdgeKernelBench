# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: depthwise_pointwise_bn_act.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_068_depthwise_pointwise_bn_act',
 'level': 2,
 'name': 'depthwise_pointwise_bn_act',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'normalization'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py',
                          'symbol': 'Conv2dNormActivation'}],
 'summary': 'MobileNetV2 depthwise Conv2dNormActivation followed by pointwise Conv2dNormActivation'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, out_channels):
        super().__init__()
        self.depthwise = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False), nn.BatchNorm2d(channels), nn.ReLU6(inplace=False))
        self.pointwise = nn.Sequential(nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, out_channels]
