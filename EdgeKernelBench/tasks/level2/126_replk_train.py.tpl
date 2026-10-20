# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: replk_train.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_126_replk_train',
 'level': 2,
 'name': 'replk_train',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'activation', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/replknet.py'}],
 'summary': 'Source: https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/replknet.py (RepLKBlock).',
 'adaptation_notes': ['unreparameterized training graph: parallel depthwise 31x31 and 5x5 branches plus '
                      'identity, then pointwise mixing and residual.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.large = nn.Conv2d(channels, channels, 31, padding=15, groups=channels, bias=False)
        self.small = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.pw1 = nn.Conv2d(channels, channels, 1)
        self.pw2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.large(x) + self.small(x)
        return x + self.pw2(F.gelu(self.pw1(y)))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels]
