# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: video_sensor_fusion.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_159_video_sensor_fusion',
 'level': 2,
 'name': 'video_sensor_fusion',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'activation',
                   'reduction',
                   'recurrent',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'video Conv2d branch and sensor GRU branch are independent until concat+Linear fusion'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, sensor_dim, hidden_size, out_dim):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(channels, hidden_size, 3, stride=2, padding=1), nn.SiLU(), nn.Conv2d(hidden_size, hidden_size, 3, stride=2, padding=1), nn.SiLU())
        self.sensor = nn.GRU(sensor_dim, hidden_size, batch_first=True)
        self.fuse = nn.Linear(hidden_size * 2, out_dim)

    def forward(self, frame: torch.Tensor, sensor_seq: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        v = self.vision(frame).mean(dim=(-2, -1))
        s, _ = self.sensor(sensor_seq, h0)
        return self.fuse(torch.cat((v, s[:, 79, :]), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
hidden_size = __EKB_PARAM_hidden_size__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
seq_len = __EKB_PARAM_seq_len__
sensor_dim = __EKB_PARAM_sensor_dim__

def get_inputs():
    frame = torch.rand(batch_size, channels, height, width)
    sensor_seq = torch.rand(batch_size, seq_len, sensor_dim)
    h0 = torch.rand(__EKB_INPUT_get_inputs_001_c001__, batch_size, hidden_size)
    return [frame, sensor_seq, h0]

def get_init_inputs():
    return [channels, sensor_dim, hidden_size, out_dim]
