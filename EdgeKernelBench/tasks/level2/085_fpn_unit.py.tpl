# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: fpn_unit.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_085_fpn_unit',
 'level': 2,
 'name': 'fpn_unit',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'resampling', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py',
                          'symbol': 'FeaturePyramidNetwork'}],
 'summary': 'two-level FPN: inner 1x1 convs, nearest top-down upsample, lateral add, 3x3 output convs'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, low_channels, high_channels, out_channels):
        super().__init__()
        self.inner_low = nn.Conv2d(low_channels, out_channels, kernel_size=1, padding=0)
        self.inner_high = nn.Conv2d(high_channels, out_channels, kernel_size=1, padding=0)
        self.layer_low = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.layer_high = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        last_inner = self.inner_high(high)
        high_out = self.layer_high(last_inner)
        inner_lateral = self.inner_low(low)
        top_down = F.interpolate(last_inner, size=(height, width), mode='nearest')
        last_inner = inner_lateral + top_down
        low_out = self.layer_low(last_inner)
        return (low_out, high_out)
batch_size = __EKB_PARAM_batch_size__
low_channels = __EKB_PARAM_low_channels__
high_channels = __EKB_PARAM_high_channels__
fpn_channels = __EKB_PARAM_fpn_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, low_channels, height, width), torch.rand(batch_size, high_channels, height // __EKB_INPUT_get_inputs_001_c001__, width // __EKB_INPUT_get_inputs_001_c002__)]

def get_init_inputs():
    return [low_channels, high_channels, fpn_channels]
