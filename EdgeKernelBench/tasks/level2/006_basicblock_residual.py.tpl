# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/pytorch--vision/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: basicblock_residual.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_006_basicblock_residual',
 'level': 2,
 'name': 'basicblock_residual',
 'scenario_tags': ['mobile_personal', 'embodied_robotics'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'residual', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py',
                          'symbol': 'BasicBlock'}],
 'summary': 'BasicBlock no-downsample path: conv3x3/bn/relu -> conv3x3/bn -> residual add -> ReLU'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels]
