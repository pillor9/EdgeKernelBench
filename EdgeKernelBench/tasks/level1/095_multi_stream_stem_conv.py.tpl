# EdgeKernelBench licensing
# SPDX-License-Identifier: AGPL-3.0-only
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/ultralytics--ultralytics/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: multi_stream_stem_conv.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_095_multi_stream_stem_conv',
 'level': 1,
 'name': 'multi_stream_stem_conv',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution', 'normalization', 'activation'],
 'semantic_references': [{'url': 'https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py',
                          'symbol': 'Conv'}],
 'summary': 'Ultralytics-style Conv stem: activation(batchnorm(conv(x))) with a 3x3 stride-2 convolution.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__
stride = __EKB_PARAM_stride__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride]
