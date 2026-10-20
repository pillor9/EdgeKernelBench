# EdgeKernelBench licensing
# SPDX-License-Identifier: AGPL-3.0-only
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/ultralytics--yolov5/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: focus_space_to_depth.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_083_focus_space_to_depth',
 'level': 2,
 'name': 'focus_space_to_depth',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'tensor_transform'],
 'semantic_references': [{'url': 'https://github.com/ultralytics/yolov5/blob/master/models/common.py',
                          'symbol': 'Focus'}],
 'summary': 'Focus slices four checkerboard spatial offsets, concatenates them on channels, then applies '
            'Conv'}
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

class Model(nn.Module):

    def __init__(self, channels, out_channels):
        super().__init__()
        self.conv = Conv(channels * 4, out_channels, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1)
        return self.conv(y)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, out_channels]
