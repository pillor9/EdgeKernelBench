# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: swin_window_attention.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_154_swin_window_attention',
 'level': 2,
 'name': 'swin_window_attention',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention',
                   'matrix_multiplication',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'tensor_generation',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer.py'}],
 'summary': 'Source: '
            'https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer.py '
            '(WindowAttention).',
 'adaptation_notes': ['non-shifted static window attention with QKV, relative-position-table gather, bias '
                      'add, softmax, and output projection.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, dim: int, heads: int, window_size: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.window_size = window_size
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        side = 2 * window_size - 1
        self.relative_position_bias_table = nn.Parameter(torch.zeros(side * side, heads))
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij')).flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative[0] += window_size - 1
        relative[1] += window_size - 1
        relative[0] *= side
        self.register_buffer('relative_position_index', relative.sum(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.reshape(n, n, self.heads).permute(2, 0, 1)
        weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.head_dim ** (-0.5) + bias.unsqueeze(0), dim=-1)
        y = torch.matmul(weights, v).transpose(1, 2).reshape(b, n, d)
        return self.proj(y)
batch_size = __EKB_PARAM_batch_size__
dim = __EKB_PARAM_dim__
heads = __EKB_PARAM_heads__
window_size = __EKB_PARAM_window_size__

def get_inputs():
    return [torch.randn(batch_size, window_size * window_size, dim)]

def get_init_inputs():
    return [dim, heads, window_size]
