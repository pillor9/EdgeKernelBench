# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: tinyllama_decoder2.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_037_tinyllama_decoder2',
 'level': 3,
 'name': 'tinyllama_decoder2',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['attention',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'residual',
                   'tensor_transform',
                   'tensor_generation',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py'}],
 'summary': 'Source: '
            'https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py '
            '(LlamaDecoderLayer).',
 'adaptation_notes': ['two compact static TinyLlama decoder layers with RMSNorm, rotary GQA, causal '
                      'attention, gated SiLU MLP, and residual paths; model depth is intentionally truncated '
                      'to two layers.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

class RMSNorm(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-06

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps) * self.weight

class DecoderLayer(nn.Module):

    def __init__(self, dim: int, q_heads: int, kv_heads: int, hidden: int, seq: int):
        super().__init__()
        self.qh = q_heads
        self.kvh = kv_heads
        self.hd = dim // q_heads
        self.groups = q_heads // kv_heads
        self.input_norm = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, q_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.hd, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.hd, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.post_norm = RMSNorm(dim)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        positions = torch.arange(seq, dtype=torch.float32).reshape(seq, 1)
        inv = 1.0 / 10000.0 ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd)
        freq = positions * inv.reshape(1, -1)
        emb = torch.cat((freq, freq), dim=-1)
        self.register_buffer('cos', emb.cos().reshape(1, 1, seq, self.hd), persistent=False)
        self.register_buffer('sin', emb.sin().reshape(1, 1, seq, self.hd), persistent=False)
        self.register_buffer('mask', torch.triu(torch.ones(seq, seq), diagonal=1).reshape(1, 1, seq, seq) * -10000.0, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.input_norm(x)
        b, n, d = y.shape
        q = self.q_proj(y).reshape(b, n, self.qh, self.hd).transpose(1, 2)
        k = self.k_proj(y).reshape(b, n, self.kvh, self.hd).transpose(1, 2)
        v = self.v_proj(y).reshape(b, n, self.kvh, self.hd).transpose(1, 2)
        q = q * self.cos + rotate_half(q) * self.sin
        k = k * self.cos + rotate_half(k) * self.sin
        k = k.reshape(b, self.kvh, 1, n, self.hd).expand(b, self.kvh, self.groups, n, self.hd).reshape(b, self.qh, n, self.hd)
        v = v.reshape(b, self.kvh, 1, n, self.hd).expand(b, self.kvh, self.groups, n, self.hd).reshape(b, self.qh, n, self.hd)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.hd ** (-0.5) + self.mask, dim=-1)
        x = residual + self.o_proj(torch.matmul(attn, v).transpose(1, 2).reshape(b, n, d))
        y = self.post_norm(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))

class Model(nn.Module):

    def __init__(self, dim: int, q_heads: int, kv_heads: int, hidden: int, seq: int):
        super().__init__()
        self.layer1 = DecoderLayer(dim, q_heads, kv_heads, hidden, seq)
        self.layer2 = DecoderLayer(dim, q_heads, kv_heads, hidden, seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))
batch_size = __EKB_PARAM_batch_size__
seq = __EKB_PARAM_seq__
dim = __EKB_PARAM_dim__
q_heads = __EKB_PARAM_q_heads__
kv_heads = __EKB_PARAM_kv_heads__
hidden = __EKB_PARAM_hidden__

def get_inputs():
    return [torch.randn(batch_size, seq, dim)]

def get_init_inputs():
    return [dim, q_heads, kv_heads, hidden, seq]
