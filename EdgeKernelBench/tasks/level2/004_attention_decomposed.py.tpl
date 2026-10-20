# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: attention_decomposed.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_004_attention_decomposed',
 'level': 2,
 'name': 'attention_decomposed',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention', 'matrix_multiplication', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/ScalingIntelligence/KernelBench/tree/main/KernelBench/level2',
                          'symbol': 'matmul_scale_softmax_matmul'}],
 'summary': 'scaled dot-product attention core: q @ k^T -> scale -> softmax -> @ v'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, head_dim):
        super().__init__()
        self.scale = head_dim ** (-0.5)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)
batch_size = __EKB_PARAM_batch_size__
heads = __EKB_PARAM_heads__
seq_len = __EKB_PARAM_seq_len__
head_dim = __EKB_PARAM_head_dim__

def get_inputs():
    return [torch.rand(batch_size, heads, seq_len, head_dim), torch.rand(batch_size, heads, seq_len, head_dim), torch.rand(batch_size, heads, seq_len, head_dim)]

def get_init_inputs():
    return [head_dim]
