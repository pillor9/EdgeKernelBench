# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: dense_moe.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_067_dense_moe',
 'level': 2,
 'name': 'dense_moe',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['moe', 'matrix_multiplication', 'activation', 'tensor_generation', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py'}],
 'summary': 'Source: '
            'https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py '
            '(MixtralSparseMoeBlock).',
 'adaptation_notes': ['Static rewrite computes all four experts densely and combines them with router '
                      'softmax weights.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.r = nn.Linear(dim, 4)
        self.a = nn.ModuleList([nn.Linear(dim, hidden) for _ in range(4)])
        self.b = nn.ModuleList([nn.Linear(hidden, dim) for _ in range(4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.r(x), dim=-1)
        y = self.b[0](F.silu(self.a[0](x))) * w[..., 0:1]
        y = y + self.b[1](F.silu(self.a[1](x))) * w[..., 1:2]
        y = y + self.b[2](F.silu(self.a[2](x))) * w[..., 2:3]
        return y + self.b[3](F.silu(self.a[3](x))) * w[..., 3:4]
batch_size = __EKB_PARAM_batch_size__
tokens = __EKB_PARAM_tokens__
dim = __EKB_PARAM_dim__
hidden = __EKB_PARAM_hidden__

def get_inputs():
    return [torch.randn(batch_size, tokens, dim)]

def get_init_inputs():
    return [dim, hidden]
