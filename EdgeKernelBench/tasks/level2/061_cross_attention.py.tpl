# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: cross_attention.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_061_cross_attention',
 'level': 2,
 'name': 'cross_attention',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention', 'matrix_multiplication', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py'}],
 'summary': 'Source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py '
            '(cross-attention path).',
 'adaptation_notes': ['queries are projected from x and keys/values from the independent static context '
                      'input.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, dim: int, context_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(context_dim, dim)
        self.v = nn.Linear(context_dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        m = context.shape[1]
        q = self.q(x).reshape(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(context).reshape(b, m, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(context).reshape(b, m, self.heads, self.head_dim).permute(0, 2, 1, 3)
        weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.head_dim ** (-0.5), dim=-1)
        y = torch.matmul(weights, v).permute(0, 2, 1, 3).reshape(b, n, d)
        return self.out(y)
batch_size = __EKB_PARAM_batch_size__
tokens = __EKB_PARAM_tokens__
context_tokens = __EKB_PARAM_context_tokens__
dim = __EKB_PARAM_dim__
context_dim = __EKB_PARAM_context_dim__
heads = __EKB_PARAM_heads__

def get_inputs():
    return [torch.randn(batch_size, tokens, dim), torch.randn(batch_size, context_tokens, context_dim)]

def get_init_inputs():
    return [dim, context_dim, heads]
