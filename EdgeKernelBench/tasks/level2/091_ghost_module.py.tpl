# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/huawei-noah--Efficient-AI-Backbones/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: ghost_module.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_091_ghost_module',
 'level': 2,
 'name': 'ghost_module',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py',
                          'symbol': 'GhostModule'}],
 'summary': 'GhostModule primary 1x1 conv plus cheap depthwise 3x3 conv then concat and output-channel slice'}
import math
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, ratio=2):
        super().__init__()
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        self.out_channels = out_channels
        self.primary_conv = nn.Sequential(nn.Conv2d(in_channels, init_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(init_channels), nn.ReLU(inplace=False))
        self.cheap_operation = nn.Sequential(nn.Conv2d(init_channels, new_channels, kernel_size=3, stride=1, padding=1, groups=init_channels, bias=False), nn.BatchNorm2d(new_channels), nn.ReLU(inplace=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, out_channels]
