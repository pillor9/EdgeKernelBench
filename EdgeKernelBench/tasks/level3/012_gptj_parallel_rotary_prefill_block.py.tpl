# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: gptj_parallel_rotary_prefill_block.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l3_012_gptj_parallel_rotary_prefill_block',
 'level': 3,
 'name': 'gptj_parallel_rotary_prefill_block',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'rotary_position',
                   'tensor_transform',
                   'type_conversion',
                   'tensor_generation',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/gpt_j/gpt_j_model.py',
                          'symbol': 'GPTJBlock'}],
 'summary': 'GPT-J-style MHA prefill block with rotary Q/K and parallel attention/MLP residuals.',
 'adaptation_notes': ['Scaled fp32 static adaptation preserving MHA, rotary Q/K, and the parallel residual '
                      'topology. Partial rotary dimensions, cache state, dropout, and fused attention are '
                      'omitted.'],
 'model_families': ['GPT-J']}
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

class Model(nn.Module):

    def __init__(self, dim: int, ffn_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ln_1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.fc_in = nn.Linear(dim, ffn_dim, bias=True)
        self.fc_out = nn.Linear(ffn_dim, dim, bias=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor):
        normed = self.ln_1(x)
        batch, seq_len, _ = x.shape
        q = self.q_proj(normed).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * (1.0 / math.sqrt(self.head_dim))
        scores = scores + mask
        probabilities = F.softmax(scores, dim=-1)
        context = torch.matmul(probabilities, v)
        context = context.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
        attention_output = self.out_proj(context)
        mlp_output = self.fc_out(F.gelu(self.fc_in(normed)))
        return x + attention_output + mlp_output

def get_init_inputs():
    return [__EKB_INPUT_get_init_inputs_001_c001__, __EKB_INPUT_get_init_inputs_001_c002__, __EKB_INPUT_get_init_inputs_001_c003__, __EKB_INPUT_get_init_inputs_001_c004__]

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
