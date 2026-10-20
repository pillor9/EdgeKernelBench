# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: mqa_prefill.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l2_115_mqa_prefill',
 'level': 2,
 'name': 'mqa_prefill',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['attention', 'matrix_multiplication', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/llama/llama_model.py',
                          'symbol': 'LlamaAttention'}],
 'summary': 'Causal multi-query prefill attention using one K/V head.',
 'adaptation_notes': ['The referenced configurable KV-head path is specialized to one KV head and rendered '
                      'with an explicit static mask and attention math.'],
 'model_families': ['Llama-MQA']}
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
        self.q_width = num_q_heads * head_dim
        self.kv_width = num_kv_heads * head_dim
        self.scale = head_dim ** (-0.5)
        self.qkv_proj = nn.Linear(dim, self.q_width + 2 * self.kv_width, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        packed = self.qkv_proj(x)
        q, k, v = torch.split(packed, [self.q_width, self.kv_width, self.kv_width], dim=-1)
        batch_size, query_len, _ = x.shape
        q = q.reshape(batch_size, query_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, query_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, query_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
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
    mask = torch.full((__EKB_INPUT_get_inputs_001_c002__, __EKB_INPUT_get_inputs_001_c003__, seq_len, seq_len), -__EKB_INPUT_get_inputs_001_c004__)
    mask = torch.triu(mask, diagonal=__EKB_INPUT_get_inputs_001_c005__ + (seq_len - seq_len))
    return [x, mask]

def get_init_inputs():
    return [dim, num_q_heads, num_kv_heads, head_dim]
