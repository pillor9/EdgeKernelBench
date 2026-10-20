# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: channel_shuffle.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_026_channel_shuffle',
 'level': 1,
 'name': 'channel_shuffle',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py',
                          'symbol': 'channel_shuffle'}],
 'summary': 'view(batch, groups, channels_per_group, h, w) -> transpose(1,2).contiguous() -> view(batch, '
            'channels, h, w)'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(batch_size, num_channels, height, width)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
groups = __EKB_PARAM_groups__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [groups]
