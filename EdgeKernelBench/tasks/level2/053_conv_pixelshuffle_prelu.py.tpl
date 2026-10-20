# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv_pixelshuffle_prelu.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_053_conv_pixelshuffle_prelu',
 'level': 2,
 'name': 'conv_pixelshuffle_prelu',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution', 'activation', 'tensor_transform'],
 'semantic_references': [{'url': 'https://github.com/pytorch/examples/blob/main/super_resolution/model.py',
                          'symbol': 'Net.conv4 / PixelShuffle'}],
 'summary': 'super-resolution tail: conv to upscale_factor^2 channels, PixelShuffle, PReLU'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * upscale_factor * upscale_factor, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pixel_shuffle(self.conv(x))
        return self.prelu(x)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, out_channels]
