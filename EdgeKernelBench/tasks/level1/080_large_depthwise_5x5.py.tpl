# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: large_depthwise_5x5.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_080_large_depthwise_5x5',
 'level': 1,
 'name': 'large_depthwise_5x5',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution'],
 'semantic_references': [{'url': 'https://github.com/yjn870/FSRCNN-pytorch/blob/master/models.py',
                          'symbol': 'FSRCNN first_part conv reference'}],
 'summary': 'Depthwise 5x5 convolution inspired by super-resolution feature maps.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise(x)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, kernel_size]
