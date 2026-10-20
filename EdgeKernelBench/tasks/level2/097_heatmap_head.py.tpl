# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: heatmap_head.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_097_heatmap_head',
 'level': 2,
 'name': 'heatmap_head',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/dense_heads/centerpoint_head.py',
                          'symbol': 'CenterHead / SeparateHead'}],
 'summary': 'CenterPoint shared_conv followed by SeparateHead heatmap conv stack and inference sigmoid'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, share_channels, head_channels, num_classes):
        super().__init__()
        self.shared_conv = nn.Sequential(nn.Conv2d(in_channels, share_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(share_channels), nn.ReLU(inplace=False))
        self.heatmap = nn.Sequential(nn.Conv2d(share_channels, head_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(head_channels), nn.ReLU(inplace=False), nn.Conv2d(head_channels, num_classes, kernel_size=3, padding=1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shared_conv(x)
        return torch.sigmoid(self.heatmap(x))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
mid_channels = __EKB_PARAM_mid_channels__
num_classes = __EKB_PARAM_num_classes__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, out_channels, mid_channels, num_classes]
