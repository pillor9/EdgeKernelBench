# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: inverted_residual_no_se.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_102_inverted_residual_no_se',
 'level': 2,
 'name': 'inverted_residual_no_se',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'normalization', 'residual', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py',
                          'symbol': 'InvertedResidual'}],
 'summary': 'MobileNetV2 InvertedResidual residual path: expand 1x1 -> depthwise 3x3 -> project 1x1 -> add '
            'input'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, hidden_dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=False), nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=False), nn.Conv2d(hidden_dim, channels, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
hidden_dim = __EKB_PARAM_hidden_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, hidden_dim]
