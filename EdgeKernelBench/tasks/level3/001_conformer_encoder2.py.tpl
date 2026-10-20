# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conformer_encoder2.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_001_conformer_encoder2',
 'level': 3,
 'name': 'conformer_encoder2',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['attention',
                   'convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'embedding',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/sooftware/conformer/blob/main/conformer/encoder.py'}],
 'summary': 'Source: https://github.com/sooftware/conformer/blob/main/conformer/encoder.py '
            '(ConformerEncoder).',
 'adaptation_notes': ['two static Conformer-style encoder layers, each FFN-half residual, explicit MHA, '
                      'GLU-decomposed convolution module, second FFN-half residual, and LayerNorm.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Layer(nn.Module):

    def __init__(self, dim: int, heads: int, hidden: int):
        super().__init__()
        self.h = heads
        self.d = dim // heads
        self.n1 = nn.LayerNorm(dim)
        self.f1a = nn.Linear(dim, hidden)
        self.f1b = nn.Linear(hidden, dim)
        self.n2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.n3 = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, 2 * dim, 1)
        self.dw = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.pw2 = nn.Conv1d(dim, dim, 1)
        self.n4 = nn.LayerNorm(dim)
        self.f2a = nn.Linear(dim, hidden)
        self.f2b = nn.Linear(hidden, dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.f1b(F.silu(self.f1a(self.n1(x))))
        b, n, d = x.shape
        qkv = self.qkv(self.n2(x)).reshape(b, n, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        a = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.d ** (-0.5), dim=-1)
        x = x + self.proj(torch.matmul(a, v).transpose(1, 2).reshape(b, n, d))
        y = self.pw1(self.n3(x).transpose(1, 2))
        a, b2 = torch.chunk(y, 2, dim=1)
        y = a * torch.sigmoid(b2)
        y = self.pw2(F.silu(self.bn(self.dw(y)))).transpose(1, 2)
        x = x + y
        return self.out_norm(x + 0.5 * self.f2b(F.silu(self.f2a(self.n4(x)))))

class Model(nn.Module):

    def __init__(self, dim: int, heads: int, hidden: int, vocab: int):
        super().__init__()
        self.layers = nn.Sequential(Layer(dim, heads, hidden), Layer(dim, heads, hidden))
        self.head = nn.Linear(dim, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.layers(x))
batch_size = __EKB_PARAM_batch_size__
tokens = __EKB_PARAM_tokens__
dim = __EKB_PARAM_dim__
heads = __EKB_PARAM_heads__
hidden = __EKB_PARAM_hidden__
vocab = __EKB_PARAM_vocab__

def get_inputs():
    return [torch.randn(batch_size, tokens, dim)]

def get_init_inputs():
    return [dim, heads, hidden, vocab]
