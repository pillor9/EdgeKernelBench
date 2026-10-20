# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: multiscale_fusion.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_116_multiscale_fusion',
 'level': 2,
 'name': 'multiscale_fusion',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution', 'resampling', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py',
                          'symbol': 'FeaturePyramidNetwork'}],
 'summary': 'three-level FPN variant: inner 1x1 convs, two nearest upsample+add fusions, output 3x3 convs'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inner_low = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.inner_mid = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.inner_high = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.layer_low = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.layer_mid = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.layer_high = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, low: torch.Tensor, mid: torch.Tensor, high: torch.Tensor):
        last_inner = self.inner_high(high)
        out_high = self.layer_high(last_inner)
        inner_mid = self.inner_mid(mid)
        last_inner = inner_mid + F.interpolate(last_inner, size=(height // 2, width // 2), mode='nearest')
        out_mid = self.layer_mid(last_inner)
        inner_low = self.inner_low(low)
        last_inner = inner_low + F.interpolate(last_inner, size=(height, width), mode='nearest')
        out_low = self.layer_low(last_inner)
        return (out_low, out_mid, out_high)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    low = torch.rand(batch_size, channels, height, width)
    mid = torch.rand(batch_size, channels, height // __EKB_INPUT_get_inputs_001_c001__, width // __EKB_INPUT_get_inputs_001_c002__)
    high = torch.rand(batch_size, channels, height // __EKB_INPUT_get_inputs_001_c003__, width // __EKB_INPUT_get_inputs_001_c004__)
    return [low, mid, high]

def get_init_inputs():
    return [channels, out_channels]
