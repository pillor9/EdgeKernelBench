# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: espcn.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_008_espcn',
 'level': 3,
 'name': 'espcn',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution', 'activation', 'tensor_transform'],
 'semantic_references': [{'url': 'https://github.com/pytorch/examples/blob/main/super_resolution/model.py'}],
 'summary': 'Source: https://github.com/pytorch/examples/blob/main/super_resolution/model.py (Net).',
 'adaptation_notes': ['full four-convolution ESPCN feature chain followed by sub-pixel PixelShuffle '
                      'reconstruction.']}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels: int, c1: int, c2: int, c3: int, scale: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, c1, 5, padding=2)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.conv4 = nn.Conv2d(c3, channels * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return self.shuffle(self.conv4(x))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
c1 = __EKB_PARAM_c1__
c2 = __EKB_PARAM_c2__
c3 = __EKB_PARAM_c3__
scale = __EKB_PARAM_scale__

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, c1, c2, c3, scale]
