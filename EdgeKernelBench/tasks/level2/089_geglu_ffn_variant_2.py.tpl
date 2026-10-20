# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: geglu_ffn_variant_2.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l2_089_geglu_ffn_variant_2',
 'level': 2,
 'name': 'geglu_ffn_variant_2',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['matrix_multiplication', 'activation', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/gemma2/gemma2_model.py',
                          'symbol': 'Gemma2MLP'}],
 'summary': 'Prefill GeGLU gate/up/down feed-forward sublayer.',
 'adaptation_notes': ['The gate activation is specialized to GELU and expressed as separate dense '
                      'projections and elementwise multiplication.'],
 'model_families': ['Gemma-style GeGLU']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(hidden)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__
ffn_dim = __EKB_PARAM_ffn_dim__

def get_inputs():
    torch.manual_seed(__EKB_INPUT_get_inputs_001_c001__)
    return [torch.randn(batch_size, seq_len, dim, dtype=torch.float32)]

def get_init_inputs():
    return [dim, ffn_dim]
