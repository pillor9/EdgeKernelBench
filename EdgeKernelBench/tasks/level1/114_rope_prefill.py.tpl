# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: rope_prefill.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l1_114_rope_prefill',
 'level': 1,
 'name': 'rope_prefill',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['rotary_position', 'tensor_transform', 'type_conversion', 'tensor_generation'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/qwen2/qwen2_model.py',
                          'symbol': 'QWen2Attention'}],
 'summary': 'Apply rotary-position arithmetic to every query head in a prefill sequence.',
 'adaptation_notes': ['Materialize fixed fp32 cosine/sine tables and use portable rotate-half arithmetic.'],
 'model_families': ['Qwen2']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    paired = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1)
    return rotated.reshape_as(x)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin

class Model(nn.Module):

    def forward(self, query: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return apply_rope(query, cos, sin)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
num_heads = __EKB_PARAM_num_heads__
head_dim = __EKB_PARAM_head_dim__

def get_inputs():
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    positions = torch.arange(seq_len, dtype=torch.float32)
    inv_freq = __EKB_INPUT_get_inputs_001_c001__ / __EKB_INPUT_get_inputs_001_c002__ ** (torch.arange(__EKB_INPUT_get_inputs_001_c003__, head_dim, __EKB_INPUT_get_inputs_001_c004__).float() / head_dim)
    freqs = torch.outer(positions, inv_freq)
    angles = torch.stack((freqs, freqs), dim=-__EKB_INPUT_get_inputs_001_c005__).reshape(seq_len, head_dim)
    cos = torch.cos(angles).reshape(__EKB_INPUT_get_inputs_001_c006__, __EKB_INPUT_get_inputs_001_c007__, seq_len, head_dim)
    sin = torch.sin(angles).reshape(__EKB_INPUT_get_inputs_001_c008__, __EKB_INPUT_get_inputs_001_c009__, seq_len, head_dim)
    return [query, cos, sin]

def get_init_inputs():
    return []
