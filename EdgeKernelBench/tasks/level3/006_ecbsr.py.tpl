# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: ecbsr.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_006_ecbsr',
 'level': 3,
 'name': 'ecbsr',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution', 'activation', 'resampling', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/xindongzhang/ECBSR'}],
 'summary': 'Source: https://github.com/xindongzhang/ECBSR (ECB/ECBSR).',
 'adaptation_notes': ['Compact ECBSR-inspired workload with two blocks, each summing learned 3x3, 1x1, '
                      'depthwise 3x3, and identity branches before ReLU. Pixel-shuffle reconstruction '
                      'is added to a nearest-neighbor upsampled input.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECB(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.k3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.k1 = nn.Conv2d(channels, channels, 1)
        self.edge = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.k3(x) + self.k1(x) + self.edge(x))

class Model(nn.Module):

    def __init__(self, width: int):
        super().__init__()
        self.head = nn.Conv2d(3, width, 3, padding=1)
        self.body = nn.Sequential(ECB(width), ECB(width))
        self.tail = nn.Sequential(nn.Conv2d(width, 12, 3, padding=1), nn.PixelShuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.body(self.head(x))) + F.interpolate(x, scale_factor=2, mode='nearest')
batch_size = __EKB_PARAM_batch_size__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
channels = __EKB_PARAM_channels__

def get_inputs():
    return [torch.randn(batch_size, __EKB_INPUT_get_inputs_001_c001__, height, width)]

def get_init_inputs():
    return [channels]
