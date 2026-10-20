# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: gpt2_mha_gelu_prefill_block.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l3_011_gpt2_mha_gelu_prefill_block',
 'level': 3,
 'name': 'gpt2_mha_gelu_prefill_block',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['attention',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/gpt2/gpt2_model.py',
                          'symbol': 'GPT2Block'}],
 'summary': 'Static-sequence GPT-2-style prefill block with fused-QKV MHA and GELU MLP.',
 'adaptation_notes': ['Scaled fp32 block translation with embedded hidden states as input. Absolute-position '
                      'embedding lookup, dropout, cache objects, and optimized attention are outside the '
                      'case; causal MHA is expressed as explicit matmul-mask-softmax-matmul.'],
 'model_families': ['GPT-2']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, dim: int, ffn_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ln_1 = nn.LayerNorm(dim)
        self.c_attn = nn.Linear(dim, 3 * dim, bias=True)
        self.c_proj = nn.Linear(dim, dim, bias=True)
        self.ln_2 = nn.LayerNorm(dim)
        self.c_fc = nn.Linear(dim, ffn_dim, bias=True)
        self.c_mlp_proj = nn.Linear(ffn_dim, dim, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        normed = self.ln_1(x)
        q, k, v = torch.split(self.c_attn(normed), x.shape[-1], dim=-1)
        batch, seq_len, _ = x.shape
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * (1.0 / math.sqrt(self.head_dim))
        scores = scores + mask
        probabilities = F.softmax(scores, dim=-1)
        context = torch.matmul(probabilities, v)
        context = context.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
        hidden = x + self.c_proj(context)
        return hidden + self.c_mlp_proj(F.gelu(self.c_fc(self.ln_2(hidden))))

def get_init_inputs():
    return [__EKB_INPUT_get_init_inputs_001_c001__, __EKB_INPUT_get_init_inputs_001_c002__, __EKB_INPUT_get_init_inputs_001_c003__, __EKB_INPUT_get_init_inputs_001_c004__]

def get_inputs():
    torch.manual_seed(__EKB_INPUT_get_inputs_001_c001__)
    batch, seq_len, dim, head_dim = (__EKB_INPUT_get_inputs_001_c002__, __EKB_INPUT_get_inputs_001_c003__, __EKB_INPUT_get_inputs_001_c004__, __EKB_INPUT_get_inputs_001_c005__)
    x = torch.randn(batch, seq_len, dim, dtype=torch.float32)
    mask = torch.full((__EKB_INPUT_get_inputs_001_c006__, __EKB_INPUT_get_inputs_001_c007__, seq_len, seq_len), -__EKB_INPUT_get_inputs_001_c008__)
    mask = torch.triu(mask, diagonal=__EKB_INPUT_get_inputs_001_c009__ + (seq_len - seq_len))
    return [x, mask]
