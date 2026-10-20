# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: sdpa_long_sequence.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_116_sdpa_long_sequence',
 'level': 1,
 'name': 'sdpa_long_sequence',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['attention', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py',
                          'symbol': 'Attention'}],
 'summary': 'timm attention unfused path computes q*scale -> q @ k.transpose(-2,-1) -> softmax -> attn @ v'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, head_dim: int):
        super().__init__()
        self.scale = head_dim ** (-0.5)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attn = q * self.scale @ k.transpose(-2, -1)
        attn = F.softmax(attn, dim=-1)
        return attn @ v
batch_size = __EKB_PARAM_batch_size__
num_heads = __EKB_PARAM_num_heads__
seq_len = __EKB_PARAM_seq_len__
head_dim = __EKB_PARAM_head_dim__

def get_inputs():
    q = torch.rand(batch_size, num_heads, seq_len, head_dim)
    k = torch.rand(batch_size, num_heads, seq_len, head_dim)
    v = torch.rand(batch_size, num_heads, seq_len, head_dim)
    return [q, k, v]

def get_init_inputs():
    return [head_dim]
