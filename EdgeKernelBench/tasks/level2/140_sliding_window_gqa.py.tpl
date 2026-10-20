# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: sliding_window_gqa.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l2_140_sliding_window_gqa',
 'level': 2,
 'name': 'sliding_window_gqa',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention',
                   'matrix_multiplication',
                   'activation',
                   'rotary_position',
                   'tensor_transform',
                   'type_conversion',
                   'tensor_generation',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/mistral/mistral_model.py',
                          'symbol': 'MistralAttention'}],
 'summary': 'Rotary grouped-query prefill attention restricted to a fixed causal sliding window.',
 'adaptation_notes': ['Runtime cache windowing becomes a precomputed additive band mask; attention and KV '
                      'repetition remain ordinary PyTorch operations.'],
 'model_families': ['Mistral-style GQA']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, dim: int, num_q_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.dim = dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_width = num_q_heads * head_dim
        self.kv_width = num_kv_heads * head_dim
        self.scale = head_dim ** (-0.5)
        self.qkv_proj = nn.Linear(dim, self.q_width + 2 * self.kv_width, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        packed = self.qkv_proj(x)
        q, k, v = torch.split(packed, [self.q_width, self.kv_width, self.kv_width], dim=-1)
        batch_size, query_len, _ = x.shape
        q = q.reshape(batch_size, query_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, query_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, query_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = repeat_kv(k, self.num_q_heads, self.num_kv_heads)
        v = repeat_kv(v, self.num_q_heads, self.num_kv_heads)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        probabilities = torch.softmax(scores + mask, dim=-1)
        context = torch.matmul(probabilities, v)
        context = context.transpose(1, 2).reshape(batch_size, query_len, self.dim)
        return self.out_proj(context)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__
num_q_heads = __EKB_PARAM_num_q_heads__
num_kv_heads = __EKB_PARAM_num_kv_heads__
head_dim = __EKB_PARAM_head_dim__
window = __EKB_PARAM_window__

def get_inputs():
    torch.manual_seed(__EKB_INPUT_get_inputs_001_c001__)
    x = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)
    positions = torch.arange(seq_len, dtype=torch.float32)
    inv_freq = __EKB_INPUT_get_inputs_001_c002__ / __EKB_INPUT_get_inputs_001_c003__ ** (torch.arange(__EKB_INPUT_get_inputs_001_c004__, head_dim, __EKB_INPUT_get_inputs_001_c005__).float() / head_dim)
    freqs = torch.outer(positions, inv_freq)
    angles = torch.stack((freqs, freqs), dim=-__EKB_INPUT_get_inputs_001_c006__).reshape(seq_len, head_dim)
    cos = torch.cos(angles).reshape(__EKB_INPUT_get_inputs_001_c007__, __EKB_INPUT_get_inputs_001_c008__, seq_len, head_dim)
    sin = torch.sin(angles).reshape(__EKB_INPUT_get_inputs_001_c009__, __EKB_INPUT_get_inputs_001_c010__, seq_len, head_dim)
    query_positions = torch.arange(seq_len).reshape(seq_len, __EKB_INPUT_get_inputs_001_c011__)
    key_positions = torch.arange(seq_len).reshape(__EKB_INPUT_get_inputs_001_c012__, seq_len)
    allowed = (key_positions <= query_positions) & (key_positions > query_positions - window)
    mask = torch.where(allowed, torch.zeros(seq_len, seq_len, dtype=torch.float32), torch.full((seq_len, seq_len), -__EKB_INPUT_get_inputs_001_c013__, dtype=torch.float32)).reshape(__EKB_INPUT_get_inputs_001_c014__, __EKB_INPUT_get_inputs_001_c015__, seq_len, seq_len)
    return [x, cos, sin, mask]

def get_init_inputs():
    return [dim, num_q_heads, num_kv_heads, head_dim]
