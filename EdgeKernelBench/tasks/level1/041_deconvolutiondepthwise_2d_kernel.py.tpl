# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: deconvolutiondepthwise_2d_kernel.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_041_deconvolutiondepthwise_2d_kernel',
 'level': 1,
 'name': 'deconvolutiondepthwise_2d_kernel',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Convolution/DeconvolutionDepthwise_2D_kernel.py',
                          'symbol': 'DeconvolutionDepthwise_2D_kernel'}],
 'summary': 'nn.ConvTranspose2d(groups=in_channels, kernel_size=5, stride=1, padding=2, bias=True)'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, k: int, stride, padding, output_padding, dilation):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=in_ch, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__
stride = __EKB_PARAM_stride__
padding = __EKB_PARAM_padding__
output_padding = __EKB_PARAM_output_padding__
dilation = __EKB_PARAM_dilation__

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation]
