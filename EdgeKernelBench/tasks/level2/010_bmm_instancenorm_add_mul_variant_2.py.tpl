# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/ScalingIntelligence--KernelBench/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: bmm_instancenorm_add_mul_variant_2.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_010_bmm_instancenorm_add_mul_variant_2',
 'level': 2,
 'name': 'bmm_instancenorm_add_mul_variant_2',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['matrix_multiplication', 'normalization', 'residual', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/KernelBench/level2/28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py',
                          'symbol': 'user_sample'}],
 'summary': 'Linear(x) -> InstanceNorm2d(unsqueeze(unsqueeze(x))) -> squeeze -> residual add -> multiply'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_features, out_features, eps=1e-05, momentum=0.1):
        super().__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.bmm(x)
        x = self.instance_norm(x.unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1)
        x = x + y
        x = x * y
        return x
batch_size = __EKB_PARAM_batch_size__
in_features = __EKB_PARAM_in_features__
out_features = __EKB_PARAM_out_features__

def get_inputs():
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]

def get_init_inputs():
    return [in_features, out_features]
