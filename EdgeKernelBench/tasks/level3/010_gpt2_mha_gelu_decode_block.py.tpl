# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: gpt2_mha_gelu_decode_block.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l3_010_gpt2_mha_gelu_decode_block',
 'level': 3,
 'name': 'gpt2_mha_gelu_decode_block',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['attention',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/gpt2/gpt2_model.py',
                          'symbol': 'GPT2Block'}],
 'summary': 'Single-token GPT-2-style MHA decoder block with fused QKV, LayerNorm, and GELU MLP.',
 'adaptation_notes': ['Scaled fp32 PyTorch decode translation. Token and absolute-position embedding lookup '
                      'is outside the block; its input is already embedded. Runtime cache state is '
                      'represented by explicit fixed-size rolling K/V tensors, and attention is decomposed '
                      'explicitly.'],
 'model_families': ['GPT-2']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, dim: int, ffn_dim: int, num_heads: int, head_dim: int, cache_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.cache_len = cache_len
        self.ln_1 = nn.LayerNorm(dim)
        self.c_attn = nn.Linear(dim, 3 * dim, bias=True)
        self.c_proj = nn.Linear(dim, dim, bias=True)
        self.ln_2 = nn.LayerNorm(dim)
        self.c_fc = nn.Linear(dim, ffn_dim, bias=True)
        self.c_mlp_proj = nn.Linear(ffn_dim, dim, bias=True)

    def forward(self, x: torch.Tensor, past_key: torch.Tensor, past_value: torch.Tensor, mask: torch.Tensor):
        normed = self.ln_1(x)
        q, k, v = torch.split(self.c_attn(normed), x.shape[-1], dim=-1)
        batch = x.shape[0]
        q = q.reshape(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key_all = torch.cat((past_key, k), dim=2)
        value_all = torch.cat((past_value, v), dim=2)
        new_key = key_all[:, :, 1:, :]
        new_value = value_all[:, :, 1:, :]
        scores = torch.matmul(q, key_all.transpose(-2, -1))
        scores = scores * (1.0 / math.sqrt(self.head_dim))
        scores = scores + mask
        probabilities = F.softmax(scores, dim=-1)
        context = torch.matmul(probabilities, value_all)
        context = context.transpose(1, 2).reshape(batch, 1, self.num_heads * self.head_dim)
        hidden = x + self.c_proj(context)
        hidden = hidden + self.c_mlp_proj(F.gelu(self.c_fc(self.ln_2(hidden))))
        return (hidden, new_key, new_value)

def get_init_inputs():
    return [__EKB_INPUT_get_init_inputs_001_c001__, __EKB_INPUT_get_init_inputs_001_c002__, __EKB_INPUT_get_init_inputs_001_c003__, __EKB_INPUT_get_init_inputs_001_c004__, __EKB_INPUT_get_init_inputs_001_c005__]

def get_inputs():
    torch.manual_seed(__EKB_INPUT_get_inputs_001_c001__)
    batch, dim = (__EKB_INPUT_get_inputs_001_c002__, __EKB_INPUT_get_inputs_001_c003__)
    num_kv_heads, head_dim, cache_len = (__EKB_INPUT_get_inputs_001_c004__, __EKB_INPUT_get_inputs_001_c005__, __EKB_INPUT_get_inputs_001_c006__)
    x = torch.randn(batch, __EKB_INPUT_get_inputs_001_c007__, dim, dtype=torch.float32)
    past_key = torch.randn(batch, num_kv_heads, cache_len, head_dim, dtype=torch.float32)
    past_value = torch.randn(batch, num_kv_heads, cache_len, head_dim, dtype=torch.float32)
    mask = torch.zeros(__EKB_INPUT_get_inputs_001_c008__, __EKB_INPUT_get_inputs_001_c009__, __EKB_INPUT_get_inputs_001_c010__, __EKB_INPUT_get_inputs_001_c011__, dtype=torch.float32)
    return [x, past_key, past_value, mask]
