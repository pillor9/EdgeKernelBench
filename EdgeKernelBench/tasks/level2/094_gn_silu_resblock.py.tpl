# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: gn_silu_resblock.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_094_gn_silu_resblock',
 'level': 2,
 'name': 'gn_silu_resblock',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py'}],
 'summary': 'Source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py '
            '(ResnetBlock2D).',
 'adaptation_notes': ['same-channel GroupNorm, SiLU, conv, GroupNorm, SiLU, conv, and residual path; time '
                      'embedding/dropout omitted because this static block has none.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, channels: int, groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels, eps=1e-06)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels, eps=1e-06)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.silu(self.norm1(x)))
        y = self.conv2(F.silu(self.norm2(y)))
        return x + y
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
groups = __EKB_PARAM_groups__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, groups]
