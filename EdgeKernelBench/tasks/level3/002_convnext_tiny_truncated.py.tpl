# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: convnext_tiny_truncated.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_002_convnext_tiny_truncated',
 'level': 3,
 'name': 'convnext_tiny_truncated',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py'}],
 'summary': 'Source: https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py '
            '(ConvNeXt-Tiny).',
 'adaptation_notes': ['Compact ConvNeXt-inspired classifier with a patchify stem, two CNBlocks per '
                      'stage, one downsampling step, global average, and a classifier. The stem uses '
                      'BatchNorm; the CNBlocks and downsampling path use LayerNorm.']}
import torch
import torch.nn as nn

class CNBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-06)
        self.pw1 = nn.Linear(channels, 4 * channels)
        self.pw2 = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(torch.full((channels, 1, 1), 1e-06))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x).permute(0, 2, 3, 1)
        y = self.pw2(torch.nn.functional.gelu(self.pw1(self.norm(y)))).permute(0, 3, 1, 2)
        return x + self.gamma * y

class Model(nn.Module):

    def __init__(self, stem_channels: int, stage2_channels: int, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, stem_channels, 4, stride=4), nn.BatchNorm2d(stem_channels))
        self.stage1 = nn.Sequential(CNBlock(stem_channels), CNBlock(stem_channels))
        self.downsample = nn.Sequential(nn.LayerNorm(stem_channels), nn.Conv2d(stem_channels, stage2_channels, 2, stride=2))
        self.stage2 = nn.Sequential(CNBlock(stage2_channels), CNBlock(stage2_channels))
        self.norm = nn.LayerNorm(stage2_channels)
        self.head = nn.Linear(stage2_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(self.stem(x))
        x = x.permute(0, 2, 3, 1)
        x = self.downsample[1](self.downsample[0](x).permute(0, 3, 1, 2))
        x = self.stage2(x).mean(dim=(2, 3))
        return self.head(self.norm(x))
batch_size = __EKB_PARAM_batch_size__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
stem_channels = __EKB_PARAM_stem_channels__
stage2_channels = __EKB_PARAM_stage2_channels__
num_classes = __EKB_PARAM_num_classes__

def get_inputs():
    return [torch.randn(batch_size, __EKB_INPUT_get_inputs_001_c001__, height, width)]

def get_init_inputs():
    return [stem_channels, stage2_channels, num_classes]
