# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conv_branch_transformer_branch.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_052_conv_branch_transformer_branch',
 'level': 2,
 'name': 'conv_branch_transformer_branch',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['attention',
                   'convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'image conv branch and token self-attention branch are independent until concat+Linear fusion'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenAttention(nn.Module):

    def __init__(self, dim, heads, head_dim, out_dim):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** (-0.5)
        self.q = nn.Linear(dim, inner, bias=False)
        self.k = nn.Linear(dim, inner, bias=False)
        self.v = nn.Linear(dim, inner, bias=False)
        self.proj = nn.Linear(inner, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        q = q.reshape(batch_size * self.heads, seq_len, self.head_dim)
        k = k.reshape(batch_size * self.heads, seq_len, self.head_dim)
        v = v.reshape(batch_size * self.heads, seq_len, self.head_dim)
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        out = torch.bmm(F.softmax(attn, dim=-1), v)
        out = out.view(batch_size, self.heads, seq_len, self.head_dim).transpose(1, 2).reshape(batch_size, seq_len, self.heads * self.head_dim)
        return self.proj(out).mean(dim=1)

class Model(nn.Module):

    def __init__(self, channels, dim, heads, head_dim, mid_channels, out_dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(channels, mid_channels, 3, stride=2, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU())
        self.attn = TokenAttention(dim, heads, head_dim, out_dim)
        self.fuse = nn.Linear(mid_channels + out_dim, out_dim)

    def forward(self, image: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        conv_feat = self.conv(image).mean(dim=(-2, -1))
        token_feat = self.attn(tokens)
        return self.fuse(torch.cat((conv_feat, token_feat), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
dim = __EKB_PARAM_dim__
heads = __EKB_PARAM_heads__
head_dim = __EKB_PARAM_head_dim__
mid_channels = __EKB_PARAM_mid_channels__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
seq_len = __EKB_PARAM_seq_len__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width), torch.rand(batch_size, seq_len, dim)]

def get_init_inputs():
    return [channels, dim, heads, head_dim, mid_channels, out_dim]
