# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: c2f_chunk_dual_branch.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_013_c2f_chunk_dual_branch',
 'level': 2,
 'name': 'c2f_chunk_dual_branch',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [],
 'summary': 'C2f-style cv1 output is split into two independent bottleneck branches and fused by concat+cv2'}
import torch
import torch.nn as nn

class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, (k - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class Model(nn.Module):

    def __init__(self, channels, out_channels):
        super().__init__()
        hidden = out_channels // 2
        self.cv1 = Conv(channels, 2 * hidden, 1, 1)
        self.branch0 = nn.Sequential(Conv(hidden, hidden, 3, 1), Conv(hidden, hidden, 3, 1))
        self.branch1 = nn.Sequential(Conv(hidden, hidden, 3, 1), Conv(hidden, hidden, 3, 1))
        self.cv2 = Conv(4 * hidden, out_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0, y1 = self.cv1(x).chunk(2, 1)
        b0 = self.branch0(y0)
        b1 = self.branch1(y1)
        return self.cv2(torch.cat((y0, y1, b0, b1), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, out_channels]
