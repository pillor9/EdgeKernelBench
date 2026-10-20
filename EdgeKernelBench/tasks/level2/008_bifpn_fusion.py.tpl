# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: bifpn_fusion.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_008_bifpn_fusion',
 'level': 2,
 'name': 'bifpn_fusion',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'activation', 'reduction', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/efficientdet/model.py'}],
 'summary': 'Source: '
            'https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/efficientdet/model.py '
            '(BiFPN).',
 'adaptation_notes': ['Two-input nonnegative learned weighted fusion, normalization by weight sum, '
                      'depthwise 3x3 convolution, then SiLU.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2))
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weights = F.relu(self.weights)
        fused = (weights[0] * x + weights[1] * y) / (weights.sum() + 0.0001)
        return F.silu(self.conv(fused))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.randn(batch_size, channels, height, width), torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels]
