# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: prefill_norm_swiglu_residual.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l2_125_prefill_norm_swiglu_residual',
 'level': 2,
 'name': 'prefill_norm_swiglu_residual',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'residual',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/llama/llama_model.py',
                          'symbol': 'LlamaDecoderLayer'}],
 'summary': 'Prefill RMSNorm, SwiGLU FFN, and residual addition.',
 'adaptation_notes': ['The compact subgraph omits tensor-parallel communication and uses only traceable FP32 '
                      'PyTorch operations.'],
 'model_families': ['Llama']}
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

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(x)
        hidden = F.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        return x + self.down_proj(hidden)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__
ffn_dim = __EKB_PARAM_ffn_dim__

def get_inputs():
    torch.manual_seed(__EKB_INPUT_get_inputs_001_c001__)
    return [torch.randn(batch_size, seq_len, dim, dtype=torch.float32)]

def get_init_inputs():
    return [dim, ffn_dim]
