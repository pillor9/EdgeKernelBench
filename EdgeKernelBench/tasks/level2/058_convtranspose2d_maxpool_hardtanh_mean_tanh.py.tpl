# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/ScalingIntelligence--KernelBench/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: convtranspose2d_maxpool_hardtanh_mean_tanh.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_058_convtranspose2d_maxpool_hardtanh_mean_tanh',
 'level': 2,
 'name': 'convtranspose2d_maxpool_hardtanh_mean_tanh',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'activation', 'pooling', 'reduction', 'tensor_transform'],
 'semantic_references': [{'url': 'https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh.py',
                          'symbol': 'Model'}],
 'summary': 'ConvTranspose2d -> MaxPool2d -> Hardtanh -> mean over H,W -> tanh'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = self.hardtanh(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = torch.tanh(x)
        return x
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__
stride = __EKB_PARAM_stride__
padding = __EKB_PARAM_padding__
maxpool_kernel_size = __EKB_PARAM_maxpool_kernel_size__
maxpool_stride = __EKB_PARAM_maxpool_stride__
hardtanh_min = __EKB_PARAM_hardtanh_min__
hardtanh_max = __EKB_PARAM_hardtanh_max__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max]
