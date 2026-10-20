# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: convtranspose2d_mish_add_hardtanh_scaling.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_059_convtranspose2d_mish_add_hardtanh_scaling',
 'level': 2,
 'name': 'convtranspose2d_mish_add_hardtanh_scaling',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling.py',
                          'symbol': 'Model'}],
 'summary': 'ConvTranspose2d -> Mish -> add scalar -> Hardtanh -> scalar multiply'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = x * torch.tanh(F.softplus(x))
        x = x + self.add_value
        x = F.hardtanh(x, min_val=-1, max_val=1)
        x = x * self.scale
        return x
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__
stride = __EKB_PARAM_stride__
padding = __EKB_PARAM_padding__
output_padding = __EKB_PARAM_output_padding__
add_value = __EKB_PARAM_add_value__
scale = __EKB_PARAM_scale__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]
