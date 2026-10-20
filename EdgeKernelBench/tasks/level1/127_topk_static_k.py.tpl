# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: topk_static_k.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_127_topk_static_k',
 'level': 1,
 'name': 'topk_static_k',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['selection'],
 'semantic_references': [{'url': 'https://github.com/facebookresearch/detr', 'symbol': 'torch.topk'}],
 'summary': 'Return the largest k values along each input row; k and the input dimensions come from the profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, k: int=300):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, _ = torch.topk(x, k=self.k, dim=1)
        return values
batch_size = __EKB_PARAM_batch_size__
dim = __EKB_PARAM_dim__
k = __EKB_PARAM_k__

def get_inputs():
    return [torch.rand(batch_size, dim)]

def get_init_inputs():
    return [k]
