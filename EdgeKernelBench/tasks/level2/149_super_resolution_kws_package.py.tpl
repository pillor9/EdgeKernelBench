# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: super_resolution_kws_package.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_149_super_resolution_kws_package',
 'level': 2,
 'name': 'super_resolution_kws_package',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'super-resolution Conv+PixelShuffle branch and DS-CNN audio branch run independently before '
            'vector fusion'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, mid_channels, out_channels, out_dim):
        super().__init__()
        self.sr = nn.Sequential(nn.Conv2d(channels, mid_channels, 5, padding=2), nn.Tanh(), nn.Conv2d(mid_channels, out_channels * 4, 3, padding=1), nn.PixelShuffle(2), nn.PReLU(out_channels))
        self.kws = nn.Sequential(nn.Conv2d(1, mid_channels, 3, padding=1, groups=1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(), nn.Conv2d(mid_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.fuse = nn.Linear(out_channels * 2, out_dim)

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        sr_feat = self.sr(image).mean(dim=(-2, -1))
        kws_feat = self.kws(audio).mean(dim=(-2, -1))
        return self.fuse(torch.cat((sr_feat, kws_feat), dim=1))
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
