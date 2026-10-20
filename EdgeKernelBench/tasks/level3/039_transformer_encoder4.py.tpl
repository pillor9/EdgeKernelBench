# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: transformer_encoder4.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_039_transformer_encoder4',
 'level': 3,
 'name': 'transformer_encoder4',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py'}],
 'summary': 'Source: '
            'https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py '
            '(Block stack).',
 'adaptation_notes': ['four static pre-norm transformer encoder blocks with explicit QKV attention and GELU '
                      'MLP, followed by LayerNorm.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):

    def __init__(self, d: int, h: int):
        super().__init__()
        self.h = h
        self.hd = d // h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.p = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, 4 * d)
        self.f2 = nn.Linear(4 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(self.n1(x)).reshape(b, n, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        a = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.hd ** (-0.5), dim=-1)
        x = x + self.p(torch.matmul(a, v).transpose(1, 2).reshape(b, n, d))
        return x + self.f2(F.gelu(self.f1(self.n2(x))))

class Model(nn.Module):

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.blocks = nn.Sequential(Block(dim, heads), Block(dim, heads), Block(dim, heads), Block(dim, heads))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.blocks(x))
batch_size = __EKB_PARAM_batch_size__
tokens = __EKB_PARAM_tokens__
dim = __EKB_PARAM_dim__
heads = __EKB_PARAM_heads__

def get_inputs():
    return [torch.randn(batch_size, tokens, dim)]

def get_init_inputs():
    return [dim, heads]
