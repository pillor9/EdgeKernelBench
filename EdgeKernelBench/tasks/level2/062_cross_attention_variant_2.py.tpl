# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: cross_attention_variant_2.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l2_062_cross_attention_variant_2',
 'level': 2,
 'name': 'cross_attention_variant_2',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention', 'matrix_multiplication', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/llama/llama_model.py',
                          'symbol': 'LlamaAttention'}],
 'summary': 'Grouped-query attention from a prefill query stream to a separate memory stream.',
 'adaptation_notes': ['Self-attention projections are recast as stateless cross-attention with explicit '
                      'query and memory tensors and no cache object.'],
 'model_families': ['Llama-style cross attention']}
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
        self.q_proj = nn.Linear(dim, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, query_len, _ = query.shape
        memory_len = memory.shape[1]
        q = self.q_proj(query).reshape(batch_size, query_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).reshape(batch_size, memory_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).reshape(batch_size, memory_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = repeat_kv(k, self.num_q_heads, self.num_kv_heads)
        v = repeat_kv(v, self.num_q_heads, self.num_kv_heads)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        probabilities = torch.softmax(scores + mask, dim=-1)
        context = torch.matmul(probabilities, v)
        context = context.transpose(1, 2).reshape(batch_size, query_len, self.dim)
        return self.out_proj(context)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
context_len = __EKB_PARAM_context_len__
dim = __EKB_PARAM_dim__
num_q_heads = __EKB_PARAM_num_q_heads__
num_kv_heads = __EKB_PARAM_num_kv_heads__
head_dim = __EKB_PARAM_head_dim__

def get_inputs():
    torch.manual_seed(__EKB_INPUT_get_inputs_001_c001__)
    query = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)
    memory = torch.randn(batch_size, context_len, dim, dtype=torch.float32)
    mask = torch.zeros(__EKB_INPUT_get_inputs_001_c002__, __EKB_INPUT_get_inputs_001_c003__, seq_len, context_len, dtype=torch.float32)
    return [query, memory, mask]

def get_init_inputs():
    return [dim, num_q_heads, num_kv_heads, head_dim]
