# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: espcn_dscnn_parallel.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_078_espcn_dscnn_parallel',
 'level': 2,
 'name': 'espcn_dscnn_parallel',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'ESPCN super-resolution feature branch and DS-CNN audio branch are independent until '
            'concat+Linear fusion'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, mid_channels, out_channels, out_dim):
        super().__init__()
        self.espcn = nn.Sequential(nn.Conv2d(channels, mid_channels, 5, padding=2), nn.Tanh(), nn.Conv2d(mid_channels, mid_channels, 3, padding=1), nn.Tanh(), nn.Conv2d(mid_channels, out_channels * 4, 3, padding=1), nn.PixelShuffle(2))
        self.dscnn = nn.Sequential(nn.Conv2d(1, mid_channels, 3, padding=1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(), nn.Conv2d(mid_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.fuse = nn.Linear(out_channels * 2, out_dim)

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        a = self.espcn(image).mean(dim=(-2, -1))
        b = self.dscnn(audio).mean(dim=(-2, -1))
        return self.fuse(torch.cat((a, b), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
mid_channels = __EKB_PARAM_mid_channels__
out_channels = __EKB_PARAM_out_channels__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width), torch.rand(batch_size, __EKB_INPUT_get_inputs_001_c001__, height, width)]

def get_init_inputs():
    return [channels, mid_channels, out_channels, out_dim]
