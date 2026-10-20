# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: ddrnet_slim.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_003_ddrnet_slim',
 'level': 3,
 'name': 'ddrnet_slim',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'resampling', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/ydhongHIT/DDRNet'}],
 'summary': 'Source: https://github.com/ydhongHIT/DDRNet (DDRNet-23-slim).',
 'adaptation_notes': ['Compact dual-resolution segmentation workload with high/low branches, '
                      'bidirectional fusion, a global-average context branch, and an output head.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):

    def __init__(self, ci: int, co: int, s: int=1):
        super().__init__()
        self.c = nn.Conv2d(ci, co, 3, stride=s, padding=1, bias=False)
        self.b = nn.BatchNorm2d(co)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.b(self.c(x)))

class Model(nn.Module):

    def __init__(self, high: int, low: int, classes: int):
        super().__init__()
        self.stem = nn.Sequential(Conv(3, high, 2), Conv(high, high, 2))
        self.high = Conv(high, high)
        self.down = Conv(high, low, 2)
        self.low = Conv(low, low)
        self.h2l = Conv(high, low, 2)
        self.l2h = nn.Conv2d(low, high, 1)
        self.context = nn.Conv2d(low, low, 1)
        self.head = nn.Sequential(Conv(high, high), nn.Conv2d(high, classes, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.high(self.stem(x))
        l = self.low(self.down(h))
        l = l + self.h2l(h)
        h = h + F.interpolate(self.l2h(l), size=h.shape[-2:], mode='bilinear', align_corners=False)
        ctx = self.context(F.adaptive_avg_pool2d(l, 1))
        h = h + F.interpolate(self.l2h(ctx), size=h.shape[-2:], mode='nearest')
        return F.interpolate(self.head(h), size=x.shape[-2:], mode='bilinear', align_corners=False)
batch_size = __EKB_PARAM_batch_size__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
high = __EKB_PARAM_high__
low = __EKB_PARAM_low__
classes = __EKB_PARAM_classes__

def get_inputs():
    return [torch.randn(batch_size, __EKB_INPUT_get_inputs_001_c001__, height, width)]

def get_init_inputs():
    return [high, low, classes]
