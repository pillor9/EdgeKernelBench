# EdgeKernelBench licensing
# SPDX-License-Identifier: AGPL-3.0-only
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/ultralytics--yolov5/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv_bn_leaky_relu.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_050_conv_bn_leaky_relu',
 'level': 2,
 'name': 'conv_bn_leaky_relu',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/ultralytics/yolov5/blob/master/models/common.py',
                          'symbol': 'Conv'}],
 'summary': 'YOLO-style Conv wrapper with Conv2d(bias=False), BatchNorm2d, and LeakyReLU activation'}
import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p

class Model(nn.Module):

    def __init__(self, channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, out_channels, 3, 1, autopad(3), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, out_channels]
