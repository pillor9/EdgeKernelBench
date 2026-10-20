# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: detr_decoder_half_layer.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_070_detr_decoder_half_layer',
 'level': 2,
 'name': 'detr_decoder_half_layer',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['attention',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/facebookresearch/detr/blob/main/models/transformer.py',
                          'symbol': 'TransformerDecoderLayer'}],
 'summary': 'DETR decoder post-norm path: self-attn -> norm -> cross-attn -> norm -> FFN -> norm'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecomposedMHA(nn.Module):

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

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(query).permute(1, 0, 2).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).permute(1, 0, 2).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).permute(1, 0, 2).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.dim).permute(1, 0, 2)
        return self.out_proj(out)

class Model(nn.Module):

    def __init__(self, dim, num_heads, hidden_dim):
        super().__init__()
        self.self_attn = DecomposedMHA(dim, num_heads)
        self.cross_attn = DecomposedMHA(dim, num_heads)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, pos: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        q = tgt + query_pos
        tgt2 = self.self_attn(q, q, tgt)
        tgt = self.norm1(tgt + tgt2)
        tgt2 = self.cross_attn(tgt + query_pos, memory + pos, memory)
        tgt = self.norm2(tgt + tgt2)
        tgt2 = self.linear2(F.relu(self.linear1(tgt)))
        return self.norm3(tgt + tgt2)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__
heads = __EKB_PARAM_heads__
hidden_dim = __EKB_PARAM_hidden_dim__

def get_inputs():
    return [torch.rand(seq_len, batch_size, dim), torch.rand(seq_len, batch_size, dim), torch.rand(seq_len, batch_size, dim), torch.rand(seq_len, batch_size, dim)]

def get_init_inputs():
    return [dim, heads, hidden_dim]
