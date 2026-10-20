# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: llama_rmsnorm_qkv.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_106_llama_rmsnorm_qkv',
 'level': 2,
 'name': 'llama_rmsnorm_qkv',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['attention',
                   'matrix_multiplication',
                   'normalization',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py',
                          'symbol': 'LlamaRMSNorm / LlamaAttention'}],
 'summary': 'Llama input RMSNorm followed by q_proj, k_proj, v_proj and view/transpose into heads'}
import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class Model(nn.Module):

    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.norm = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * head_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.norm(hidden_states)
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        return (q, k, v)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__
heads = __EKB_PARAM_heads__
head_dim = __EKB_PARAM_head_dim__

def get_inputs():
    return [torch.rand(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, heads, head_dim]
