# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: espcn_feature_chain.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_079_espcn_feature_chain',
 'level': 2,
 'name': 'espcn_feature_chain',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution', 'activation'],
 'semantic_references': [{'url': 'https://github.com/pytorch/examples/blob/main/super_resolution/model.py',
                          'symbol': 'Net'}],
 'summary': 'ESPCN feature extraction segment: 5x5 conv then 3x3 conv chain with tanh activations'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        return torch.tanh(self.conv3(x))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
mid_channels = __EKB_PARAM_mid_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, mid_channels, out_channels]
