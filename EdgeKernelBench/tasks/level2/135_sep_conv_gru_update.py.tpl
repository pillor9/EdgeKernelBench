# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/princeton-vl--RAFT/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: sep_conv_gru_update.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_135_sep_conv_gru_update',
 'level': 2,
 'name': 'sep_conv_gru_update',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution', 'activation', 'recurrent', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/princeton-vl/RAFT/blob/master/core/update.py',
                          'symbol': 'SepConvGRU'}],
 'summary': 'RAFT SepConvGRU horizontal 1x5 gates then vertical 5x1 gates'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1.0 - z) * h + z * q
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1.0 - z) * h + z * q
        return h
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    h = torch.rand(batch_size, channels, height, width)
    x = torch.rand(batch_size, out_channels, height, width)
    return [h, x]

def get_init_inputs():
    return [channels, out_channels]
