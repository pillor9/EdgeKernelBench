# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv_ae_decoder.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_047_conv_ae_decoder',
 'level': 2,
 'name': 'conv_ae_decoder',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution', 'normalization', 'activation'],
 'semantic_references': [{'url': 'https://github.com/openvinotoolkit/anomalib',
                          'symbol': 'autoencoder_decoder'}],
 'summary': 'simplified convolutional autoencoder decoder with two transposed-conv bn relu upsampling stages'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, mid_channels, out_channels):
        super().__init__()
        self.decoder = nn.Sequential(nn.ConvTranspose2d(channels, mid_channels, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(), nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
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
