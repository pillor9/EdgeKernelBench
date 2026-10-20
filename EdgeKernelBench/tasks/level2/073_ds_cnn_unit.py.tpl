# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: ds_cnn_unit.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_073_ds_cnn_unit',
 'level': 2,
 'name': 'ds_cnn_unit',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution', 'normalization', 'activation'],
 'semantic_references': [{'url': 'https://github.com/castorini/honk/blob/master/utils/model.py',
                          'symbol': 'DSCNN'}],
 'summary': 'DS-CNN unit: depthwise conv, pointwise conv, BatchNorm, ReLU'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=False), nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, out_channels]
