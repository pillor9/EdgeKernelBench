# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: minigpt_static.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_015_minigpt_static',
 'level': 3,
 'name': 'minigpt_static',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'embedding',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/ScalingIntelligence/KernelBench/tree/main/KernelBench/level3'}],
 'summary': 'Source: https://github.com/ScalingIntelligence/KernelBench/tree/main/KernelBench/level3 '
            '(MiniGPT-style static model).',
 'adaptation_notes': ['Static token/position embeddings, two causal decoder blocks with a fixed '
                      'additive mask, final normalization, and an independently parameterized '
                      'vocabulary projection.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):

    def __init__(self, d: int, h: int, seq: int):
        super().__init__()
        self.h = h
        self.hd = d // h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.p = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, 4 * d)
        self.f2 = nn.Linear(4 * d, d)
        self.register_buffer('mask', torch.triu(torch.ones(seq, seq), diagonal=1) * -10000.0, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(self.n1(x)).reshape(b, n, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        a = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.hd ** (-0.5) + self.mask, dim=-1)
        x = x + self.p(torch.matmul(a, v).transpose(1, 2).reshape(b, n, d))
        return x + self.f2(F.gelu(self.f1(self.n2(x))))

class Model(nn.Module):

    def __init__(self, vocab: int, seq: int, dim: int, heads: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, seq, dim))
        self.b1 = Block(dim, heads, seq)
        self.b2 = Block(dim, heads, seq)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.tok(ids) + self.pos
        return self.head(self.norm(self.b2(self.b1(x))))
batch_size = __EKB_PARAM_batch_size__
vocab = __EKB_PARAM_vocab__
seq = __EKB_PARAM_seq__
dim = __EKB_PARAM_dim__
heads = __EKB_PARAM_heads__

def get_inputs():
    return [torch.randint(__EKB_INPUT_get_inputs_001_c001__, vocab, (batch_size, seq))]

def get_init_inputs():
    return [vocab, seq, dim, heads]
