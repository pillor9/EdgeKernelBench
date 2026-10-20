# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: detr_mini.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_005_detr_mini',
 'level': 3,
 'name': 'detr_mini',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['attention',
                   'convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/facebookresearch/detr/blob/main/models/transformer.py'}],
 'summary': 'Source: https://github.com/facebookresearch/detr/blob/main/models/transformer.py (Transformer).',
 'adaptation_notes': ['static mini-DETR with convolutional backbone, two encoder layers, two decoder layers, '
                      'fixed learned queries, class and box heads; matching and postprocessing are '
                      'excluded.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.d = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, qx: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        b, n, d = qx.shape
        m = kv.shape[1]
        q = self.q(qx).reshape(b, n, self.h, self.d).transpose(1, 2)
        k = self.k(kv).reshape(b, m, self.h, self.d).transpose(1, 2)
        v = self.v(kv).reshape(b, m, self.h, self.d).transpose(1, 2)
        a = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.d ** (-0.5), dim=-1)
        return self.o(torch.matmul(a, v).transpose(1, 2).reshape(b, n, d))

class Encoder(nn.Module):

    def __init__(self, d: int, h: int):
        super().__init__()
        self.a = Attention(d, h)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, 4 * d)
        self.f2 = nn.Linear(4 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.n1(x + self.a(x, x))
        return self.n2(x + self.f2(F.relu(self.f1(x))))

class Decoder(nn.Module):

    def __init__(self, d: int, h: int):
        super().__init__()
        self.sa = Attention(d, h)
        self.ca = Attention(d, h)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.n3 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, 4 * d)
        self.f2 = nn.Linear(4 * d, d)

    def forward(self, q: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        q = self.n1(q + self.sa(q, q))
        q = self.n2(q + self.ca(q, m))
        return self.n3(q + self.f2(F.relu(self.f1(q))))

class Model(nn.Module):

    def __init__(self, dim: int, heads: int, queries: int, classes: int):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, dim // 2, 5, stride=2, padding=2), nn.ReLU(), nn.Conv2d(dim // 2, dim, 3, stride=2, padding=1), nn.ReLU())
        self.e1 = Encoder(dim, heads)
        self.e2 = Encoder(dim, heads)
        self.d1 = Decoder(dim, heads)
        self.d2 = Decoder(dim, heads)
        self.query = nn.Parameter(torch.randn(1, queries, dim))
        self.cls = nn.Linear(dim, classes)
        self.box = nn.Linear(dim, 4)

    def forward(self, x: torch.Tensor):
        m = self.backbone(x).flatten(2).transpose(1, 2)
        m = self.e2(self.e1(m))
        q = self.query.expand(x.shape[0], -1, -1)
        q = self.d2(self.d1(q, m), m)
        return (self.cls(q), torch.sigmoid(self.box(q)))
batch_size = __EKB_PARAM_batch_size__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
dim = __EKB_PARAM_dim__
heads = __EKB_PARAM_heads__
queries = __EKB_PARAM_queries__
classes = __EKB_PARAM_classes__

def get_inputs():
    return [torch.randn(batch_size, __EKB_INPUT_get_inputs_001_c001__, height, width)]

def get_init_inputs():
    return [dim, heads, queries, classes]
