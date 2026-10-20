# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: c3_fc_ae_variant.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_015_c3_fc_ae_variant',
 'level': 2,
 'name': 'c3_fc_ae_variant',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'C3-style conv branch and FC autoencoder branch are independent until concat+Linear fusion'}
import torch
import torch.nn as nn

class Conv(nn.Module):

    def __init__(self, c1, c2, k=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, 1, (k - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class Model(nn.Module):

    def __init__(self, channels, dim, mid_channels, out_dim):
        super().__init__()
        self.cv1 = Conv(channels, mid_channels, 1)
        self.cv2 = Conv(channels, mid_channels, 1)
        self.cv3 = Conv(mid_channels * 2, mid_channels, 1)
        self.ae = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 8), nn.ReLU(), nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, mid_channels))
        self.fuse = nn.Linear(mid_channels * 2, out_dim)

    def forward(self, image: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        c3 = self.cv3(torch.cat((self.cv1(image), self.cv2(image)), dim=1)).mean(dim=(-2, -1))
        ae = self.ae(vector)
        return self.fuse(torch.cat((c3, ae), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
dim = __EKB_PARAM_dim__
mid_channels = __EKB_PARAM_mid_channels__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width), torch.rand(batch_size, dim)]

def get_init_inputs():
    return [channels, dim, mid_channels, out_dim]
