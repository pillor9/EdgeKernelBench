# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: dense_moe_decoder.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_004_dense_moe_decoder',
 'level': 3,
 'name': 'dense_moe_decoder',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['attention',
                   'moe',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'tensor_transform',
                   'tensor_generation',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py'}],
 'summary': 'Source: '
            'https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py '
            '(MixtralSparseMoeBlock).',
 'adaptation_notes': ['static decoder self-attention followed by dense all-expert execution and '
                      'router-softmax weighted aggregation; this is a dense static MoE surrogate.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, dim: int, heads: int, hidden: int, experts: int):
        super().__init__()
        self.h = heads
        self.d = dim // heads
        self.n1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.router = nn.Linear(dim, experts)
        self.w1 = nn.ModuleList([nn.Linear(dim, hidden) for _ in range(experts)])
        self.w2 = nn.ModuleList([nn.Linear(hidden, dim) for _ in range(experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(self.n1(x)).reshape(b, n, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        mask = torch.triu(torch.ones(n, n, device=x.device), diagonal=1) * -10000.0
        a = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.d ** (-0.5) + mask, dim=-1)
        x = x + self.proj(torch.matmul(a, v).transpose(1, 2).reshape(b, n, d))
        y = self.n2(x)
        weights = F.softmax(self.router(y), dim=-1)
        output = self.w2[0](F.silu(self.w1[0](y))) * weights[..., 0:1]
        output = output + self.w2[1](F.silu(self.w1[1](y))) * weights[..., 1:2]
        output = output + self.w2[2](F.silu(self.w1[2](y))) * weights[..., 2:3]
        output = output + self.w2[3](F.silu(self.w1[3](y))) * weights[..., 3:4]
        return x + output
batch_size = __EKB_PARAM_batch_size__
tokens = __EKB_PARAM_tokens__
dim = __EKB_PARAM_dim__
heads = __EKB_PARAM_heads__
hidden = __EKB_PARAM_hidden__
experts = __EKB_PARAM_experts__

def get_inputs():
    return [torch.randn(batch_size, tokens, dim)]

def get_init_inputs():
    return [dim, heads, hidden, experts]
