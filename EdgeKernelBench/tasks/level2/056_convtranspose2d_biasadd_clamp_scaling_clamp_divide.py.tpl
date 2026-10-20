# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/ScalingIntelligence--KernelBench/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: convtranspose2d_biasadd_clamp_scaling_clamp_divide.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_056_convtranspose2d_biasadd_clamp_scaling_clamp_divide',
 'level': 2,
 'name': 'convtranspose2d_biasadd_clamp_scaling_clamp_divide',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide.py',
                          'symbol': 'Model'}],
 'summary': 'ConvTranspose2d -> bias add -> clamp -> scale -> clamp -> divide'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = x + self.bias
        x = torch.clamp(x, min=0.0, max=1.0)
        x = x * self.scaling_factor
        x = torch.clamp(x, min=0.0, max=1.0)
        x = x / self.scaling_factor
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
bias_shape = __EKB_PARAM_bias_shape__
scaling_factor = __EKB_PARAM_scaling_factor__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]
