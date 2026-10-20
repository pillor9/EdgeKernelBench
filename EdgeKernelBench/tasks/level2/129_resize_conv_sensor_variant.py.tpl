# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: resize_conv_sensor_variant.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_129_resize_conv_sensor_variant',
 'level': 2,
 'name': 'resize_conv_sensor_variant',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'resampling',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'bilinear resize plus Conv preprocessing branch and sensor MLP branch are independent until '
            'concat+Linear fusion'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, channels, sensor_dim, mid_channels, out_dim):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(channels, mid_channels, 3, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU())
        self.sensor = nn.Sequential(nn.Linear(sensor_dim, mid_channels), nn.ReLU(), nn.Linear(mid_channels, mid_channels), nn.ReLU())
        self.fuse = nn.Linear(mid_channels * 2, out_dim)

    def forward(self, image: torch.Tensor, sensor_vec: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image, size=(height + 24, width + 32), mode='bilinear', align_corners=False)
        v = self.vision(image).mean(dim=(-2, -1))
        s = self.sensor(sensor_vec)
        return self.fuse(torch.cat((v, s), dim=1))
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
