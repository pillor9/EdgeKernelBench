# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: mha.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_112_mha',
 'level': 2,
 'name': 'mha',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention', 'matrix_multiplication', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py'}],
 'summary': 'Source: '
            'https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py '
            '(Attention).',
 'adaptation_notes': ['explicit fused QKV projection, head reshape, scaled dot-product attention, and output '
                      'projection.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.head_dim ** (-0.5), dim=-1)
        y = torch.matmul(weights, v).transpose(1, 2).reshape(b, n, d)
        return self.proj(y)
batch_size = __EKB_PARAM_batch_size__
tokens = __EKB_PARAM_tokens__
dim = __EKB_PARAM_dim__
heads = __EKB_PARAM_heads__

def get_inputs():
    return [torch.randn(batch_size, tokens, dim)]

def get_init_inputs():
    return [dim, heads]
