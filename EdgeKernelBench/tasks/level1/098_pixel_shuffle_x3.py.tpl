# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: pixel_shuffle_x3.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_098_pixel_shuffle_x3',
 'level': 1,
 'name': 'pixel_shuffle_x3',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['tensor_transform'],
 'semantic_references': [{'url': 'https://github.com/pytorch/examples/blob/main/super_resolution/model.py',
                          'symbol': 'Net.pixel_shuffle'}],
 'summary': 'Sub-pixel rearrangement through nn.PixelShuffle with the scale supplied by the profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, upscale_factor: int):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pixel_shuffle(x)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
upscale_factor = __EKB_PARAM_upscale_factor__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [upscale_factor]
