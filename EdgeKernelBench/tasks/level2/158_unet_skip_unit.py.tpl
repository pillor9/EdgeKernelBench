# EdgeKernelBench licensing
# SPDX-License-Identifier: GPL-3.0-only
# Adapted from milesial/Pytorch-UNet, unet/unet_parts.py (Up).
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/milesial--Pytorch-UNet/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: unet_skip_unit.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_158_unet_skip_unit',
 'level': 2,
 'name': 'unet_skip_unit',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'residual', 'tensor_transform'],
 'semantic_references': [{'url': 'https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py',
                          'symbol': 'Up'}],
 'summary': 'U-Net Up non-bilinear branch: ConvTranspose2d upsample -> concat skip -> DoubleConv'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height // __EKB_INPUT_get_inputs_001_c001__, width // __EKB_INPUT_get_inputs_001_c002__), torch.rand(batch_size, in_channels // __EKB_INPUT_get_inputs_001_c003__, height, width)]

def get_init_inputs():
    return [in_channels, out_channels]
