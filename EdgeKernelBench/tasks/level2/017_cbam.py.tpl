# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: cbam.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_017_cbam',
 'level': 2,
 'name': 'cbam',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention', 'convolution', 'activation', 'reduction', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py'}],
 'summary': 'Source: https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py (CBAM).',
 'adaptation_notes': ['channel attention from average/max pooled descriptors followed by spatial attention '
                      'from channel mean/max maps.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        self.mlp1 = nn.Conv2d(channels, reduction, 1, bias=False)
        self.mlp2 = nn.Conv2d(reduction, channels, 1, bias=False)
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        b, c, _, _ = x.shape
        maximum = torch.max(x.reshape(b, c, -1), dim=-1, keepdim=True)[0].reshape(b, c, 1, 1)
        channel = torch.sigmoid(self.mlp2(F.relu(self.mlp1(avg))) + self.mlp2(F.relu(self.mlp1(maximum))))
        x = x * channel
        spatial_max = torch.max(x, dim=1, keepdim=True)[0]
        spatial = torch.cat([x.mean(dim=1, keepdim=True), spatial_max], dim=1)
        return x * torch.sigmoid(self.spatial(spatial))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
reduction = __EKB_PARAM_reduction__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, reduction]
