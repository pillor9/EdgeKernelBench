# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: shufflenetv2_unit.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_136_shufflenetv2_unit',
 'level': 2,
 'name': 'shufflenetv2_unit',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'tensor_transform'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py',
                          'symbol': 'InvertedResidual'}],
 'summary': 'ShuffleNetV2 stride-2 unit with branch1, branch2, concat, channel_shuffle'}
import torch
import torch.nn as nn

def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    x = x.view(batch_size, groups, out_channels // groups, height // 2, width // 2)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batch_size, out_channels, height // 2, width // 2)

class Model(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_features = out_channels // 2
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False), nn.BatchNorm2d(in_channels), nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=False))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=False), nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=2, padding=1, groups=branch_features, bias=False), nn.BatchNorm2d(branch_features), nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, out_channels]
