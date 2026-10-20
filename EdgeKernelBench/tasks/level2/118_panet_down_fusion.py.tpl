# EdgeKernelBench licensing
# SPDX-License-Identifier: AGPL-3.0-only
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/ultralytics--ultralytics/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: panet_down_fusion.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_118_panet_down_fusion',
 'level': 2,
 'name': 'panet_down_fusion',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'residual', 'tensor_transform'],
 'semantic_references': [{'url': 'https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py',
                          'symbol': 'PANet neck downsample fusion via Conv + Concat + C2f'}],
 'summary': 'YOLO PAN/FPN down path applies stride-2 Conv to a high-resolution tensor, concatenates with the '
            'lateral tensor, then applies C2f'}
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

    def __init__(self, c1, c2, k=1, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.cv1 = Conv(channels, channels, 3, 1)
        self.cv2 = Conv(channels, channels, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))

class C2f(nn.Module):

    def __init__(self, channels, out_channels):
        super().__init__()
        hidden = out_channels // 2
        self.cv1 = Conv(channels, 2 * hidden, 1, 1)
        self.b0 = Bottleneck(hidden)
        self.b1 = Bottleneck(hidden)
        self.cv2 = Conv(4 * hidden, out_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0, y1 = self.cv1(x).chunk(2, 1)
        y2 = self.b0(y1)
        y3 = self.b1(y2)
        return self.cv2(torch.cat((y0, y1, y2, y3), 1))

class Model(nn.Module):

    def __init__(self, channels, skip_channels, out_channels):
        super().__init__()
        self.down = Conv(channels, skip_channels, 3, 2)
        self.fuse = C2f(skip_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.fuse(torch.cat((x, lateral), 1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
skip_channels = __EKB_PARAM_skip_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    x = torch.rand(batch_size, channels, height, width)
    lateral = torch.rand(batch_size, skip_channels, height // __EKB_INPUT_get_inputs_001_c001__, width // __EKB_INPUT_get_inputs_001_c002__)
    return [x, lateral]

def get_init_inputs():
    return [channels, skip_channels, out_channels]
