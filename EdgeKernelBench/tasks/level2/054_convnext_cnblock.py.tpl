# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: convnext_cnblock.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_054_convnext_cnblock',
 'level': 2,
 'name': 'convnext_cnblock',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'residual',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py'}],
 'summary': 'Source: https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py (CNBlock).',
 'adaptation_notes': ['Inference CNBlock with depthwise 7x7 convolution, NHWC LayerNorm, a 4x pointwise GELU '
                      'MLP, learned channel scaling, and a residual addition.']}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-06)
        self.pw1 = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(torch.full((channels, 1, 1), 1e-06))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(x).permute(0, 2, 3, 1)
        x = self.pw2(self.act(self.pw1(self.norm(x))))
        return residual + self.gamma * x.permute(0, 3, 1, 2)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels]
