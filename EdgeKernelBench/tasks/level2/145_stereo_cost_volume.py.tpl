# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: stereo_cost_volume.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_145_stereo_cost_volume',
 'level': 2,
 'name': 'stereo_cost_volume',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['tensor_transform', 'tensor_generation', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/JiaRenChang/PSMNet/blob/master/models/stackhourglass.py',
                          'symbol': 'PSMNet cost volume construction'}],
 'summary': 'PSMNet matching volume: for each disparity, shift left/right features, concatenate channels, '
            'stack over disparity'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, disparities):
        super().__init__()
        self.disparities = disparities

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        slices = []
        for disp in range(self.disparities):
            if disp == 0:
                left_disp = left
                right_disp = right
            else:
                left_disp = F.pad(left[:, :, :, disp:], (disp, 0, 0, 0))
                right_disp = F.pad(right[:, :, :, :-disp], (disp, 0, 0, 0))
            slices.append(torch.cat([left_disp, right_disp], dim=1).unsqueeze(2))
        return torch.cat(slices, dim=2).contiguous()
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
depth = __EKB_PARAM_depth__

def get_inputs():
    left = torch.rand(batch_size, channels, height, width)
    right = torch.rand(batch_size, channels, height, width)
    return [left, right]

def get_init_inputs():
    return [depth]
