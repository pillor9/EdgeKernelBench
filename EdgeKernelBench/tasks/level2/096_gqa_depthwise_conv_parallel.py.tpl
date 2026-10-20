# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: gqa_depthwise_conv_parallel.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_096_gqa_depthwise_conv_parallel',
 'level': 2,
 'name': 'gqa_depthwise_conv_parallel',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['attention',
                   'convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'GQA token attention branch and depthwise separable Conv branch run independently before '
            'concat+Linear fusion'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQAAttention(nn.Module):

    def __init__(self, dim, heads, head_dim, out_dim):
        super().__init__()
        self.heads = heads
        self.kv_heads = heads // 2
        self.groups = heads // self.kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** (-0.5)
        self.q = nn.Linear(dim, heads * head_dim, bias=False)
        self.k = nn.Linear(dim, self.kv_heads * head_dim, bias=False)
        self.v = nn.Linear(dim, self.kv_heads * head_dim, bias=False)
        self.o = nn.Linear(heads * head_dim, out_dim, bias=False)

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, None, :, :].expand(batch_size, self.kv_heads, self.groups, seq_len, self.head_dim)
        return x.reshape(batch_size, self.heads, seq_len, self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch_size, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch_size, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)
        attn = torch.matmul(q, k.transpose(2, 3)) * self.scale
        out = torch.matmul(F.softmax(attn, dim=-1), v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.heads * self.head_dim)
        return self.o(out).mean(dim=1)

class Model(nn.Module):

    def __init__(self, channels, dim, heads, head_dim, out_dim):
        super().__init__()
        self.gqa = GQAAttention(dim, heads, head_dim, out_dim)
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False), nn.BatchNorm2d(channels), nn.ReLU(), nn.Conv2d(channels, out_dim, 1, bias=False), nn.BatchNorm2d(out_dim), nn.ReLU())
        self.fuse = nn.Linear(out_dim * 2, out_dim)

    def forward(self, image: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        token_feat = self.gqa(tokens)
        image_feat = self.conv(image).mean(dim=(-2, -1))
        return self.fuse(torch.cat((token_feat, image_feat), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
dim = __EKB_PARAM_dim__
heads = __EKB_PARAM_heads__
head_dim = __EKB_PARAM_head_dim__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
seq_len = __EKB_PARAM_seq_len__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width), torch.rand(batch_size, seq_len, dim)]

def get_init_inputs():
    return [channels, dim, heads, head_dim, out_dim]
