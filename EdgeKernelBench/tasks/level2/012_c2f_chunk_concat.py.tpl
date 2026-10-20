# EdgeKernelBench licensing
# SPDX-License-Identifier: AGPL-3.0-only
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/ultralytics--ultralytics/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: c2f_chunk_concat.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_012_c2f_chunk_concat',
 'level': 2,
 'name': 'c2f_chunk_concat',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution',
                   'normalization',
                   'activation',
                   'tensor_transform',
                   'tensor_generation',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py',
                          'symbol': 'C2f'}],
 'summary': 'C2f computes cv1, chunk into two tensors, runs Bottleneck blocks on the last branch, '
            'concatenates all branches, then cv2'}
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

    def __init__(self, channels, shortcut=False):
        super().__init__()
        self.cv1 = Conv(channels, channels, 3, 1)
        self.cv2 = Conv(channels, channels, 3, 1)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class Model(nn.Module):

    def __init__(self, channels, out_channels, n=2):
        super().__init__()
        hidden = out_channels // 2
        self.cv1 = Conv(channels, 2 * hidden, 1, 1)
        self.blocks = nn.ModuleList((Bottleneck(hidden, shortcut=False) for _ in range(n)))
        self.cv2 = Conv((2 + n) * hidden, out_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0, y1 = self.cv1(x).chunk(2, 1)
        y2 = self.blocks[0](y1)
        y3 = self.blocks[1](y2)
        return self.cv2(torch.cat((y0, y1, y2, y3), 1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, out_channels]
