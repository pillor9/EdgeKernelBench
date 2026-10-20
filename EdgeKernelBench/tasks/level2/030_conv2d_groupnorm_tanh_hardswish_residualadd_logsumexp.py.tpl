# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/ScalingIntelligence--KernelBench/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv2d_groupnorm_tanh_hardswish_residualadd_logsumexp.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_030_conv2d_groupnorm_tanh_hardswish_residualadd_logsumexp',
 'level': 2,
 'name': 'conv2d_groupnorm_tanh_hardswish_residualadd_logsumexp',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'reduction', 'residual', 'elementwise'],
 'semantic_references': [{'url': 'https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp.py',
                          'symbol': 'Model'}],
 'summary': 'Conv2d residual branch plus GroupNorm -> Tanh -> Hardswish -> residual add -> logsumexp(dim=1, '
            'keepdim=True)'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-05):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = self.conv(x)
        x_norm = self.group_norm(x_conv)
        x_tanh = self.tanh(x_norm)
        x_hard_swish = self.hard_swish(x_tanh)
        x_res = x_conv + x_hard_swish
        x_max = torch.max(x_res, dim=1, keepdim=True)[0]
        x_logsumexp = x_max + torch.log(torch.sum(torch.exp(x_res - x_max), dim=1, keepdim=True))
        return x_logsumexp
batch_size = __EKB_PARAM_batch_size__
in_channels = __EKB_PARAM_in_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
kernel_size = __EKB_PARAM_kernel_size__
groups = __EKB_PARAM_groups__

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]
