# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: imdn_block.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_100_imdn_block',
 'level': 2,
 'name': 'imdn_block',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/Zheng222/IMDN/blob/master/model/block.py'},
                         {'url': 'https://arxiv.org/abs/1909.11856',
                          'section': 'Section 3.2, Table 1, Equation (3)'}],
 'summary': 'Compact four-stage feature-distillation workload following the IMDN computation pattern, '
            'with CCA omitted.',
 'adaptation_notes': ['Quarter-channel distillation, progressive convolutions, distilled-feature '
                      'concatenation, 1x1 fusion, and residual addition.',
                      'Uses LeakyReLU with slope 0.05 after all four convolution stages; omits '
                      'contrast-aware channel attention.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, c: int):
        super().__init__()
        d = c // 4
        r = 3 * c // 4
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(r, c, 3, padding=1)
        self.c3 = nn.Conv2d(r, c, 3, padding=1)
        self.c4 = nn.Conv2d(r, d, 3, padding=1)
        self.f = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.leaky_relu(self.c1(x), 0.05)
        d1, r1 = torch.split(y, [y.shape[1] // 4, 3 * y.shape[1] // 4], 1)
        y = F.leaky_relu(self.c2(r1), 0.05)
        d2, r2 = torch.split(y, [y.shape[1] // 4, 3 * y.shape[1] // 4], 1)
        y = F.leaky_relu(self.c3(r2), 0.05)
        d3, r3 = torch.split(y, [y.shape[1] // 4, 3 * y.shape[1] // 4], 1)
        d4 = F.leaky_relu(self.c4(r3), 0.05)
        return x + self.f(torch.cat([d1, d2, d3, d4], 1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels]
