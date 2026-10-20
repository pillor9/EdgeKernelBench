# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: channel_shuffle_unit.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_019_channel_shuffle_unit',
 'level': 2,
 'name': 'channel_shuffle_unit',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py',
                          'symbol': 'channel_shuffle'}],
 'summary': 'channel_shuffle reshapes to batch x groups x channels_per_group x H x W, transposes group and '
            'channel axes, then flattens back'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, groups=4):
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(batch_size, self.groups, channels // self.groups, height, width)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, channels, height, width)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return []
