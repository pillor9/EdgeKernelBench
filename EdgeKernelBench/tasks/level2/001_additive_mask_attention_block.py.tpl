# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: additive_mask_attention_block.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_001_additive_mask_attention_block',
 'level': 2,
 'name': 'additive_mask_attention_block',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['attention', 'matrix_multiplication', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/facebookresearch/detr/blob/main/models/transformer.py',
                          'symbol': 'MultiheadAttention'}],
 'summary': 'DETR attention block with q/k/v projections, additive attention mask, softmax, value matmul and '
            'output projection'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(q_in).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_in).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_in).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        return self.out_proj(out)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__
heads = __EKB_PARAM_heads__

def get_inputs():
    return [torch.rand(batch_size, seq_len, dim), torch.rand(batch_size, seq_len, dim), torch.rand(batch_size, seq_len, dim), torch.rand(batch_size, heads, seq_len, seq_len) * -__EKB_INPUT_get_inputs_001_c001__]

def get_init_inputs():
    return [dim, heads]
