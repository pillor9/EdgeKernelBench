# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: additive_mask_attention.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_003_additive_mask_attention',
 'level': 1,
 'name': 'additive_mask_attention',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['attention', 'matrix_multiplication', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/facebookresearch/detr/blob/main/models/transformer.py',
                          'symbol': 'TransformerDecoderLayer self_attn with attn_mask'}],
 'summary': 'nn.MultiheadAttention core with attn_mask decomposed as matmul -> scale -> add(mask) -> softmax '
            '-> matmul'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, head_dim: int):
        super().__init__()
        self.scale = head_dim ** (-0.5)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        scores = q @ k.transpose(-2, -1) * self.scale
        scores = scores + attn_mask
        attn = F.softmax(scores, dim=-1)
        return attn @ v
batch_size = __EKB_PARAM_batch_size__
num_heads = __EKB_PARAM_num_heads__
seq_len = __EKB_PARAM_seq_len__
head_dim = __EKB_PARAM_head_dim__

def get_inputs():
    q = torch.rand(batch_size, num_heads, seq_len, head_dim)
    k = torch.rand(batch_size, num_heads, seq_len, head_dim)
    v = torch.rand(batch_size, num_heads, seq_len, head_dim)
    attn_mask = torch.zeros(batch_size, num_heads, seq_len, seq_len)
    return [q, k, v, attn_mask]

def get_init_inputs():
    return [head_dim]
