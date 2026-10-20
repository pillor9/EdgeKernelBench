# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: rotary_position_block.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_134_rotary_position_block',
 'level': 2,
 'name': 'rotary_position_block',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['rotary_position', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py',
                          'symbol': 'apply_rotary_pos_emb'}],
 'summary': 'apply_rotary_pos_emb(q, k, cos, sin): broadcast cos/sin, rotate_half, elementwise combine'}
import torch
import torch.nn as nn

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    return torch.cat([-x2, x1], dim=-1)

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_embed = q * cos + rotate_half(q) * sin
        k_embed = k * cos + rotate_half(k) * sin
        return (q_embed, k_embed)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
heads = __EKB_PARAM_heads__
head_dim = __EKB_PARAM_head_dim__

def get_inputs():
    q = torch.rand(batch_size, heads, seq_len, head_dim)
    k = torch.rand(batch_size, heads, seq_len, head_dim)
    cos = torch.rand(batch_size, seq_len, head_dim)
    sin = torch.rand(batch_size, seq_len, head_dim)
    return [q, k, cos, sin]

def get_init_inputs():
    return []
