# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: prefill_norm_packed_qkv.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l2_124_prefill_norm_packed_qkv',
 'level': 2,
 'name': 'prefill_norm_packed_qkv',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention', 'matrix_multiplication', 'normalization', 'reduction', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/qwen2/qwen2_model.py',
                          'symbol': 'QWen2Attention'}],
 'summary': 'Prefill RMSNorm followed by one packed grouped-query QKV projection.',
 'adaptation_notes': ['Bias and tensor-parallel dispatch are omitted; the full prefill sequence is '
                      'statically shaped and FP32.'],
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

class Model(nn.Module):

    def __init__(self, dim: int, packed_width: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.qkv_proj = nn.Linear(dim, packed_width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qkv_proj(self.norm(x))
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__
packed_width = __EKB_PARAM_packed_width__

def get_inputs():
    torch.manual_seed(__EKB_INPUT_get_inputs_001_c001__)
    return [torch.randn(batch_size, seq_len, dim, dtype=torch.float32)]

def get_init_inputs():
    return [dim, packed_width]
