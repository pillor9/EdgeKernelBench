# EdgeKernelBench licensing
# SPDX-License-Identifier: AGPL-3.0-only
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/ultralytics--ultralytics/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: detect_head_static_decode.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_069_detect_head_static_decode',
 'level': 2,
 'name': 'detect_head_static_decode',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'tensor_generation',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py',
                          'symbol': 'Detect'}],
 'summary': 'static Detect head cut point: box/class conv heads, DFL expectation, anchor-grid distance '
            'decode, class sigmoid'}
import math
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

class DWConv(Conv):

    def __init__(self, c1, c2, k=1, s=1, d=1):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d)

class DFL(nn.Module):

    def __init__(self, reg_max):
        super().__init__()
        self.reg_max = reg_max
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False).requires_grad_(False)
        bins = torch.arange(reg_max, dtype=torch.float32)
        self.conv.weight.data[:] = nn.Parameter(bins.view(1, reg_max, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(batch_size, 4, self.reg_max, num_anchors).transpose(2, 1)
        return self.conv(torch.softmax(x, dim=1)).view(batch_size, 4, num_anchors)

class Model(nn.Module):

    def __init__(self, channels, num_classes, reg_max):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        box_hidden = max(16, channels // 4, reg_max * 4)
        cls_hidden = max(channels, min(num_classes, 100))
        self.box_head = nn.Sequential(Conv(channels, box_hidden, 3), Conv(box_hidden, box_hidden, 3), nn.Conv2d(box_hidden, 4 * reg_max, 1))
        self.cls_head = nn.Sequential(nn.Sequential(DWConv(channels, channels, 3), Conv(channels, cls_hidden, 1)), nn.Sequential(DWConv(cls_hidden, cls_hidden, 3), Conv(cls_hidden, cls_hidden, 1)), nn.Conv2d(cls_hidden, num_classes, 1))
        self.dfl = DFL(reg_max)
        yy = torch.arange(height, dtype=torch.float32)
        xx = torch.arange(width, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        anchors = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), 0) + 0.5
        self.register_buffer('anchors', anchors)
        self.register_buffer('stride', torch.tensor(8.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        box_logits = self.box_head(x).view(batch_size, 4 * self.reg_max, num_anchors)
        cls_logits = self.cls_head(x).view(batch_size, self.num_classes, num_anchors)
        distances = self.dfl(box_logits)
        left_top = self.anchors.unsqueeze(0) - distances[:, 0:2, :]
        right_bottom = self.anchors.unsqueeze(0) + distances[:, 2:4, :]
        center = (left_top + right_bottom) * 0.5
        wh = right_bottom - left_top
        boxes = torch.cat((center, wh), 1) * self.stride
        return torch.cat((boxes, torch.sigmoid(cls_logits)), 1)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
num_classes = __EKB_PARAM_num_classes__
reg_max = __EKB_PARAM_reg_max__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
num_anchors = __EKB_PARAM_num_anchors__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, num_classes, reg_max]
