# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: qwen2_biased_gqa_prefill_block.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l3_028_qwen2_biased_gqa_prefill_block',
 'level': 3,
 'name': 'qwen2_biased_gqa_prefill_block',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'rotary_position',
                   'tensor_transform',
                   'type_conversion',
                   'tensor_generation',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/qwen2/qwen2_model.py',
                          'symbol': 'QWen2DecoderLayer'}],
 'summary': 'Static-sequence Qwen2-style prefill block with biased fused-QKV GQA.',
 'adaptation_notes': ['Scaled fp32 stateless translation retaining biased fused QKV, RoPE, GQA, SwiGLU, and '
                      'residuals. Runtime cache objects and fused attention are removed.'],
 'model_families': ['Qwen2']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    paired = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1)
    return rotated.reshape_as(x)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin

def repeat_kv(x: torch.Tensor, num_q_heads: int, num_kv_heads: int) -> torch.Tensor:
    groups = num_q_heads // num_kv_heads
    batch, _, tokens, head_dim = x.shape
    return x.reshape(batch, num_kv_heads, 1, tokens, head_dim).expand(batch, num_kv_heads, groups, tokens, head_dim).reshape(batch, num_q_heads, tokens, head_dim)

class Model(nn.Module):

    def __init__(self, dim: int, ffn_dim: int, num_q_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.q_heads = num_q_heads
        self.kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.ffn_dim = ffn_dim
        self.attn_norm = RMSNorm(dim)
        kv_dim = num_kv_heads * head_dim
        self.qkv_proj = nn.Linear(dim, dim + 2 * kv_dim, bias=True)
        self.o_proj = nn.Linear(num_q_heads * head_dim, dim, bias=False)
        self.ffn_norm = RMSNorm(dim)
        self.gate_up_proj = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor):
        normed = self.attn_norm(x)
        qkv = self.qkv_proj(normed)
        q_size = self.q_heads * self.head_dim
        kv_size = self.kv_heads * self.head_dim
        q, k, v = torch.split(qkv, (q_size, kv_size, kv_size), dim=-1)
        batch, seq_len, _ = x.shape
        q = q.reshape(batch, seq_len, self.q_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = repeat_kv(k, self.q_heads, self.kv_heads)
        v = repeat_kv(v, self.q_heads, self.kv_heads)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * (1.0 / math.sqrt(self.head_dim))
        scores = scores + mask
        probabilities = F.softmax(scores, dim=-1)
        context = torch.matmul(probabilities, v)
        context = context.transpose(1, 2).reshape(batch, seq_len, self.q_heads * self.head_dim)
        hidden = x + self.o_proj(context)
        ffn_input = self.ffn_norm(hidden)
        gate, up = torch.split(self.gate_up_proj(ffn_input), self.ffn_dim, dim=-1)
        return hidden + self.down_proj(F.silu(gate) * up)

def get_init_inputs():
    return [__EKB_INPUT_get_init_inputs_001_c001__, __EKB_INPUT_get_init_inputs_001_c002__, __EKB_INPUT_get_init_inputs_001_c003__, __EKB_INPUT_get_init_inputs_001_c004__, __EKB_INPUT_get_init_inputs_001_c005__]

def get_inputs():
    torch.manual_seed(__EKB_INPUT_get_inputs_001_c001__)
    batch, seq_len, dim, head_dim = (__EKB_INPUT_get_inputs_001_c002__, __EKB_INPUT_get_inputs_001_c003__, __EKB_INPUT_get_inputs_001_c004__, __EKB_INPUT_get_inputs_001_c005__)
    x = torch.randn(batch, seq_len, dim, dtype=torch.float32)
    positions = torch.arange(seq_len, dtype=torch.float32)
    inv_freq = __EKB_INPUT_get_inputs_001_c006__ / __EKB_INPUT_get_inputs_001_c007__ ** (torch.arange(__EKB_INPUT_get_inputs_001_c008__, head_dim, __EKB_INPUT_get_inputs_001_c009__).float() / head_dim)
    freqs = torch.outer(positions, inv_freq)
    angles = torch.stack((freqs, freqs), dim=-__EKB_INPUT_get_inputs_001_c010__).reshape(seq_len, head_dim)
    cos = torch.cos(angles).reshape(__EKB_INPUT_get_inputs_001_c011__, __EKB_INPUT_get_inputs_001_c012__, seq_len, head_dim)
    sin = torch.sin(angles).reshape(__EKB_INPUT_get_inputs_001_c013__, __EKB_INPUT_get_inputs_001_c014__, seq_len, head_dim)
    mask = torch.full((__EKB_INPUT_get_inputs_001_c015__, __EKB_INPUT_get_inputs_001_c016__, seq_len, seq_len), -__EKB_INPUT_get_inputs_001_c017__)
    mask = torch.triu(mask, diagonal=__EKB_INPUT_get_inputs_001_c018__ + (seq_len - seq_len))
    return [x, cos, sin, mask]
