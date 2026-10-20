# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: decode_norm_qkv_rope.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l2_064_decode_norm_qkv_rope',
 'level': 2,
 'name': 'decode_norm_qkv_rope',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['attention',
                   'matrix_multiplication',
                   'normalization',
                   'reduction',
                   'rotary_position',
                   'tensor_transform',
                   'type_conversion',
                   'tensor_generation',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/qwen2/qwen2_model.py',
                          'symbol': 'QWen2Attention'}],
 'summary': 'Single-token RMSNorm, packed QKV projection, head reshape, and rotary transform.',
 'adaptation_notes': ['Paged-cache state is removed; full-width FP32 cos/sin tables are explicit inputs.'],
 'model_families': ['Qwen2']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    paired = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1)
    return rotated.reshape_as(x)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin

class Model(nn.Module):

    def __init__(self, dim: int, num_q_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.q_width = num_q_heads * head_dim
        self.kv_width = num_kv_heads * head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.qkv_proj = nn.Linear(dim, self.q_width + 2 * self.kv_width, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packed = self.qkv_proj(self.norm(x))
        q, k, v = torch.split(packed, [self.q_width, self.kv_width, self.kv_width], dim=-1)
        batch_size = x.shape[0]
        q = q.reshape(batch_size, 1, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return (apply_rope(q, cos, sin), apply_rope(k, cos, sin), v)
batch_size = __EKB_PARAM_batch_size__
dim = __EKB_PARAM_dim__
num_q_heads = __EKB_PARAM_num_q_heads__
num_kv_heads = __EKB_PARAM_num_kv_heads__
head_dim = __EKB_PARAM_head_dim__
position = __EKB_PARAM_position__

def get_inputs():
    torch.manual_seed(__EKB_INPUT_get_inputs_001_c001__)
    x = torch.randn(batch_size, __EKB_INPUT_get_inputs_001_c002__, dim, dtype=torch.float32)
    inv_freq = __EKB_INPUT_get_inputs_001_c003__ / __EKB_INPUT_get_inputs_001_c004__ ** (torch.arange(__EKB_INPUT_get_inputs_001_c005__, head_dim, __EKB_INPUT_get_inputs_001_c006__).float() / head_dim)
    freq = float(position) * inv_freq
    angle = torch.stack((freq, freq), dim=-__EKB_INPUT_get_inputs_001_c007__).reshape(__EKB_INPUT_get_inputs_001_c008__, __EKB_INPUT_get_inputs_001_c009__, __EKB_INPUT_get_inputs_001_c010__, head_dim)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    return [x, cos, sin]

def get_init_inputs():
    return [dim, num_q_heads, num_kv_heads, head_dim]
