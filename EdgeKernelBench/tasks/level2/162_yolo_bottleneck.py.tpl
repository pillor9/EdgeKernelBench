# EdgeKernelBench licensing
# SPDX-License-Identifier: AGPL-3.0-only
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/ultralytics--ultralytics/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: yolo_bottleneck.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_162_yolo_bottleneck',
 'level': 2,
 'name': 'yolo_bottleneck',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'type_conversion', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py',
                          'symbol': 'Bottleneck'}],
 'summary': 'Ultralytics Bottleneck applies two Conv blocks and adds the shortcut when channels match'}
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

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):

    def __init__(self, channels, e=0.5):
        super().__init__()
        hidden = int(channels * e)
        self.cv1 = Conv(channels, hidden, 3, 1)
        self.cv2 = Conv(hidden, channels, 3, 1)
        self.add = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Model(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.block = Bottleneck(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels]
