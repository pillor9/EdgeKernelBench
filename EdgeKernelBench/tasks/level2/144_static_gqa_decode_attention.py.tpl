# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: static_gqa_decode_attention.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l2_144_static_gqa_decode_attention',
 'level': 2,
 'name': 'static_gqa_decode_attention',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['attention', 'matrix_multiplication', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/llama/llama_model.py',
                          'symbol': 'LlamaAttention'}],
 'summary': 'Single-token grouped-query attention over an explicit static K/V cache.',
 'adaptation_notes': ['The paged cache and fused attention call become concat, reshape-expand KV repeat, '
                      'matmul, softmax, and explicit updated-cache outputs.'],
 'model_families': ['Llama']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.scale = head_dim ** (-0.5)

    def forward(self, q: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor, current_k: torch.Tensor, current_v: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_k = torch.cat((past_k, current_k), dim=2)
        new_v = torch.cat((past_v, current_v), dim=2)
        repeated_k = repeat_kv(new_k, self.num_q_heads, self.num_kv_heads)
        repeated_v = repeat_kv(new_v, self.num_q_heads, self.num_kv_heads)
        scores = torch.matmul(q, repeated_k.transpose(-2, -1)) * self.scale
        probabilities = torch.softmax(scores + mask, dim=-1)
        context = torch.matmul(probabilities, repeated_v)
        context = context.transpose(1, 2).reshape(q.shape[0], 1, self.dim)
        return (context, new_k, new_v)
batch_size = __EKB_PARAM_batch_size__
dim = __EKB_PARAM_dim__
num_q_heads = __EKB_PARAM_num_q_heads__
num_kv_heads = __EKB_PARAM_num_kv_heads__
head_dim = __EKB_PARAM_head_dim__
cache_len = __EKB_PARAM_cache_len__

def get_inputs():
    torch.manual_seed(__EKB_INPUT_get_inputs_001_c001__)
    q = torch.randn(batch_size, num_q_heads, __EKB_INPUT_get_inputs_001_c002__, head_dim, dtype=torch.float32)
    past_k = torch.randn(batch_size, num_kv_heads, cache_len, head_dim, dtype=torch.float32)
    past_v = torch.randn(batch_size, num_kv_heads, cache_len, head_dim, dtype=torch.float32)
    current_k = torch.randn(batch_size, num_kv_heads, __EKB_INPUT_get_inputs_001_c003__, head_dim, dtype=torch.float32)
    current_v = torch.randn(batch_size, num_kv_heads, __EKB_INPUT_get_inputs_001_c004__, head_dim, dtype=torch.float32)
    mask = torch.zeros(__EKB_INPUT_get_inputs_001_c005__, __EKB_INPUT_get_inputs_001_c006__, __EKB_INPUT_get_inputs_001_c007__, cache_len + __EKB_INPUT_get_inputs_001_c008__, dtype=torch.float32)
    return [q, past_k, past_v, current_k, current_v, mask]

def get_init_inputs():
    return [dim, num_q_heads, num_kv_heads, head_dim]
