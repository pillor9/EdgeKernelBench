# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: monodepth2.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_023_monodepth2',
 'level': 3,
 'name': 'monodepth2',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution', 'activation', 'resampling', 'tensor_transform'],
 'semantic_references': [{'url': 'https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py'},
                         {'url': 'https://arxiv.org/abs/1806.01260',
                          'section': 'Section 3.3, depth network architecture'}],
 'summary': 'Compact three-scale depth encoder-decoder workload inspired by the Monodepth2 network '
            'design.',
 'adaptation_notes': ['Uses a simple convolutional encoder, zero-padded ELU decoder convolutions, '
                      'nearest upsampling, skip concatenation, and one sigmoid disparity output.',
                      'Defines a compact inference workload with no pretrained weights or training '
                      'procedure; it is not the full Monodepth2 network.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvELU(nn.Module):

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(self.conv(x))

class Model(nn.Module):

    def __init__(self, c1: int, c2: int, c3: int):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(3, c1, 7, stride=2, padding=3), nn.ELU())
        self.e2 = nn.Sequential(nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ELU())
        self.e3 = nn.Sequential(nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ELU())
        self.u2a = ConvELU(c3, c2)
        self.u2b = ConvELU(c2 + c2, c2)
        self.u1a = ConvELU(c2, c1)
        self.u1b = ConvELU(c1 + c1, c1)
        self.u0 = ConvELU(c1, c1)
        self.disp = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.e1(x)
        s2 = self.e2(s1)
        y = self.e3(s2)
        y = F.interpolate(self.u2a(y), scale_factor=2, mode='nearest')
        y = self.u2b(torch.cat([y, s2], dim=1))
        y = F.interpolate(self.u1a(y), scale_factor=2, mode='nearest')
        y = self.u1b(torch.cat([y, s1], dim=1))
        y = F.interpolate(self.u0(y), scale_factor=2, mode='nearest')
        return torch.sigmoid(self.disp(y))
batch_size = __EKB_PARAM_batch_size__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
c1 = __EKB_PARAM_c1__
c2 = __EKB_PARAM_c2__
c3 = __EKB_PARAM_c3__

def get_inputs():
    return [torch.randn(batch_size, __EKB_INPUT_get_inputs_001_c001__, height, width)]

def get_init_inputs():
    return [c1, c2, c3]
