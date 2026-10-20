# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: ae_sensor_parallel.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_002_ae_sensor_parallel',
 'level': 2,
 'name': 'ae_sensor_parallel',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'convolutional autoencoder encoder branch and sensor MLP branch are independent until '
            'concat+Linear fusion'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, sensor_dim, mid_channels, out_dim):
        super().__init__()
        self.ae = nn.Sequential(nn.Conv2d(channels, mid_channels, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(), nn.Conv2d(mid_channels, mid_channels, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU())
        self.sensor = nn.Sequential(nn.Linear(sensor_dim, mid_channels), nn.ReLU(), nn.Linear(mid_channels, mid_channels), nn.ReLU())
        self.fuse = nn.Linear(mid_channels * 2, out_dim)

    def forward(self, image: torch.Tensor, sensor_vec: torch.Tensor) -> torch.Tensor:
        a = self.ae(image).mean(dim=(-2, -1))
        s = self.sensor(sensor_vec)
        return self.fuse(torch.cat((a, s), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
mid_channels = __EKB_PARAM_mid_channels__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
sensor_dim = __EKB_PARAM_sensor_dim__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width), torch.rand(batch_size, sensor_dim)]

def get_init_inputs():
    return [channels, sensor_dim, mid_channels, out_dim]
