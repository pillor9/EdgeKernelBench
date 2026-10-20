# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: vit_tiny.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_041_vit_tiny',
 'level': 3,
 'name': 'vit_tiny',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention',
                   'convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py'}],
 'summary': 'Source: '
            'https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py '
            '(VisionTransformer tiny).',
 'adaptation_notes': ['static compact ViT with patch embedding, class token, learned position embedding, two '
                      'transformer blocks, norm and classifier.']}
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

    def __init__(self, image: int, patch: int, dim: int, heads: int, classes: int):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, patch, stride=patch)
        tokens = (image // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, tokens + 1, dim))
        self.blocks = nn.Sequential(Block(dim, heads), Block(dim, heads))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos
        return self.head(self.norm(self.blocks(x))[:, 0])
batch_size = __EKB_PARAM_batch_size__
image = __EKB_PARAM_image__
patch = __EKB_PARAM_patch__
dim = __EKB_PARAM_dim__
heads = __EKB_PARAM_heads__
classes = __EKB_PARAM_classes__

def get_inputs():
    return [torch.randn(batch_size, __EKB_INPUT_get_inputs_001_c001__, image, image)]

def get_init_inputs():
    return [image, patch, dim, heads, classes]
