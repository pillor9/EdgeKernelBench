# EdgeKernelBench licensing
# SPDX-License-Identifier: LicenseRef-NVIDIA-LSS
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
# Authors: Jonah Philion and Sanja Fidler
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/nv-tlabs--lift-splat-shoot/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: lss_depth_head.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_111_lss_depth_head',
 'level': 2,
 'name': 'lss_depth_head',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py',
                          'symbol': 'CamEncode.get_depth_feat'}],
 'summary': 'LSS depth head: depthnet 1x1 conv -> depth softmax -> depth/context outer product'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, depth_bins, context_channels):
        super().__init__()
        self.depth_bins = depth_bins
        self.context_channels = context_channels
        self.depthnet = nn.Conv2d(in_channels, depth_bins + context_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.depthnet(x)
        depth = x[:, :self.depth_bins].softmax(dim=1)
        context = x[:, self.depth_bins:self.depth_bins + self.context_channels]
        new_x = depth.unsqueeze(1) * context.unsqueeze(2)
        return (depth, new_x)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
depth_bins = __EKB_PARAM_depth_bins__
context_channels = __EKB_PARAM_context_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, depth_bins, context_channels]
