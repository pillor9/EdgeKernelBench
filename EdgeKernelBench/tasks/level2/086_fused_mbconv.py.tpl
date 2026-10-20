# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: fused_mbconv.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_086_fused_mbconv',
 'level': 2,
 'name': 'fused_mbconv',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution', 'matrix_multiplication', 'normalization', 'activation'],
 'semantic_references': [{'url': 'https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_efficientnet_blocks.py',
                          'symbol': 'EdgeResidual (FusedMBConv family)'}],
 'summary': 'EfficientNetV2-style fused MBConv: 3x3 expansion conv with activation then 1x1 linear '
            'projection'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, in_channels, expanded_channels, out_channels):
        super().__init__()
        self.conv_exp = nn.Conv2d(in_channels, expanded_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)
        self.conv_pwl = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.bn1(self.conv_exp(x)))
        return self.bn2(self.conv_pwl(x))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
hidden_dim = __EKB_PARAM_hidden_dim__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, hidden_dim, out_channels]
