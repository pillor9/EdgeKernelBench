# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv_ae_lstm_variant.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_049_conv_ae_lstm_variant',
 'level': 2,
 'name': 'conv_ae_lstm_variant',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'recurrent',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'convolutional autoencoder branch and two-layer LSTM sensor branch are independent until '
            'concat+Linear fusion'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, sensor_dim, mid_channels, out_dim):
        super().__init__()
        self.ae = nn.Sequential(nn.Conv2d(channels, mid_channels, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(), nn.ConvTranspose2d(mid_channels, mid_channels, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU())
        self.lstm = nn.LSTM(sensor_dim, mid_channels, num_layers=2, batch_first=True)
        self.fuse = nn.Linear(mid_channels * 2, out_dim)

    def forward(self, image: torch.Tensor, seq: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor) -> torch.Tensor:
        a = self.ae(image).mean(dim=(-2, -1))
        s, _ = self.lstm(seq, (h0, c0))
        return self.fuse(torch.cat((a, s[:, 79, :]), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
mid_channels = __EKB_PARAM_mid_channels__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
seq_len = __EKB_PARAM_seq_len__
sensor_dim = __EKB_PARAM_sensor_dim__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width), torch.rand(batch_size, seq_len, sensor_dim), torch.rand(__EKB_INPUT_get_inputs_001_c001__, batch_size, mid_channels), torch.rand(__EKB_INPUT_get_inputs_001_c002__, batch_size, mid_channels)]

def get_init_inputs():
    return [channels, sensor_dim, mid_channels, out_dim]
