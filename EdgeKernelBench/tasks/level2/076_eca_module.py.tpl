# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: eca_module.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_076_eca_module',
 'level': 2,
 'name': 'eca_module',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution', 'activation', 'reduction', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py',
                          'symbol': 'eca_layer'}],
 'summary': 'ECA computes global average channel descriptor, applies 1D convolution and sigmoid, then '
            'rescales the input'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean(dim=(-2, -1), keepdim=True)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels]
