# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: repvgg_train_three_branch.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_127_repvgg_train_three_branch',
 'level': 2,
 'name': 'repvgg_train_three_branch',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py',
                          'symbol': 'RepVGGBlock'}],
 'summary': 'training-mode RepVGGBlock sums 3x3 conv-bn, 1x1 conv-bn, and identity BN branches before ReLU'}
import torch
import torch.nn as nn

def conv_bn(channels, kernel_size, padding):
    return nn.Sequential(nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False), nn.BatchNorm2d(channels))

class Model(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.rbr_identity = nn.BatchNorm2d(channels)
        self.rbr_dense = conv_bn(channels, kernel_size=3, padding=1)
        self.rbr_1x1 = conv_bn(channels, kernel_size=1, padding=0)
        self.nonlinearity = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlinearity(self.rbr_dense(x) + self.rbr_1x1(x) + self.rbr_identity(x))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels]
