# EdgeKernelBench licensing
# SPDX-License-Identifier: AGPL-3.0-only
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/ultralytics--yolov5/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: c3_block.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_014_c3_block',
 'level': 2,
 'name': 'c3_block',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution',
                   'normalization',
                   'activation',
                   'tensor_transform',
                   'tensor_generation',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/ultralytics/yolov5/blob/master/models/common.py',
                          'symbol': 'C3'}],
 'summary': 'YOLOv5 C3 applies cv1 into a Bottleneck sequence, cv2 as bypass, concatenates both paths, then '
            'cv3'}
import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):

    def __init__(self, channels, shortcut=True):
        super().__init__()
        self.cv1 = Conv(channels, channels, 1, 1)
        self.cv2 = Conv(channels, channels, 3, 1)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class Model(nn.Module):

    def __init__(self, channels, out_channels, n=2):
        super().__init__()
        hidden = out_channels // 2
        self.cv1 = Conv(channels, hidden, 1, 1)
        self.cv2 = Conv(channels, hidden, 1, 1)
        self.m = nn.Sequential(*(Bottleneck(hidden, shortcut=True) for _ in range(n)))
        self.cv3 = Conv(2 * hidden, out_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, out_channels]
