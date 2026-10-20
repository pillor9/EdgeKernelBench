# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: gqa_attention_static_kv.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_095_gqa_attention_static_kv',
 'level': 2,
 'name': 'gqa_attention_static_kv',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['attention', 'matrix_multiplication', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py',
                          'symbol': 'LlamaAttention / repeat_kv / eager_attention_forward'}],
 'summary': 'Llama GQA attention with q heads and fewer kv heads, repeat_kv, scaled softmax attention, '
            'output projection'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_heads // 2
        self.num_key_value_groups = num_heads // self.num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** (-0.5)
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, None, :, :].expand(batch_size, self.num_kv_heads, self.num_key_value_groups, seq_len, self.head_dim)
        return x.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)
        attn = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__
heads = __EKB_PARAM_heads__
head_dim = __EKB_PARAM_head_dim__

def get_inputs():
    return [torch.rand(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, heads, head_dim]
