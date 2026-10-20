# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv_bn_relu_unit.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_051_conv_bn_relu_unit',
 'level': 2,
 'name': 'conv_bn_relu_unit',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'normalization', 'activation'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py',
                          'symbol': 'conv3x3/BatchNorm2d/ReLU'}],
 'summary': 'ResNet conv3x3 -> BatchNorm2d -> ReLU'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels]
