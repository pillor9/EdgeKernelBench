# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: nanodet_m.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_024_nanodet_m',
 'level': 3,
 'name': 'nanodet_m',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution',
                   'normalization',
                   'activation',
                   'resampling',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/RangiLyu/nanodet'}],
 'summary': 'Source: https://github.com/RangiLyu/nanodet (NanoDet-m backbone, PAN fusion, and heads).',
 'adaptation_notes': ['Compact NanoDet-inspired workload with a depthwise backbone, top-down '
                      'feature-pyramid fusion, and three raw prediction heads. Box decoding and NMS are '
                      'excluded.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class DS(nn.Module):

    def __init__(self, ci: int, co: int, s: int=1):
        super().__init__()
        self.d = nn.Conv2d(ci, ci, 3, stride=s, padding=1, groups=ci)
        self.p = nn.Conv2d(ci, co, 1)
        self.b = nn.BatchNorm2d(co)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.b(self.p(self.d(x))))

class Model(nn.Module):

    def __init__(self, c1: int, c2: int, c3: int, outputs: int):
        super().__init__()
        self.stem = nn.Conv2d(3, c1, 3, stride=2, padding=1)
        self.b1 = DS(c1, c1, 2)
        self.b2 = DS(c1, c2, 2)
        self.b3 = DS(c2, c3, 2)
        self.lat3 = nn.Conv2d(c3, c2, 1)
        self.f2 = DS(2 * c2, c2)
        self.lat2 = nn.Conv2d(c2, c1, 1)
        self.f1 = DS(2 * c1, c1)
        self.h1 = nn.Conv2d(c1, outputs, 1)
        self.h2 = nn.Conv2d(c2, outputs, 1)
        self.h3 = nn.Conv2d(c3, outputs, 1)

    def forward(self, x: torch.Tensor):
        p1 = self.b1(F.relu(self.stem(x)))
        p2 = self.b2(p1)
        p3 = self.b3(p2)
        u2 = F.interpolate(self.lat3(p3), size=p2.shape[-2:], mode='nearest')
        f2 = self.f2(torch.cat([p2, u2], 1))
        u1 = F.interpolate(self.lat2(f2), size=p1.shape[-2:], mode='nearest')
        f1 = self.f1(torch.cat([p1, u1], 1))
        return (self.h1(f1), self.h2(f2), self.h3(p3))
batch_size = __EKB_PARAM_batch_size__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
c1 = __EKB_PARAM_c1__
c2 = __EKB_PARAM_c2__
c3 = __EKB_PARAM_c3__
outputs = __EKB_PARAM_outputs__

def get_inputs():
    return [torch.randn(batch_size, __EKB_INPUT_get_inputs_001_c001__, height, width)]

def get_init_inputs():
    return [c1, c2, c3, outputs]
