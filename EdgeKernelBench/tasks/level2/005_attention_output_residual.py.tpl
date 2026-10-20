# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: attention_output_residual.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l2_005_attention_output_residual',
 'level': 2,
 'name': 'attention_output_residual',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention', 'matrix_multiplication', 'residual', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/llama/llama_model.py',
                          'symbol': 'LlamaDecoderLayer'}],
 'summary': 'Attention output projection followed by decoder residual addition.',
 'adaptation_notes': ['Only the output/residual tail is retained, without tensor-parallel all-reduce or '
                      'mutable state.'],
 'model_families': ['Llama']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, attention_context: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return residual + self.out_proj(attention_context)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__

def get_inputs():
    torch.manual_seed(__EKB_INPUT_get_inputs_001_c001__)
    attention_context = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)
    residual = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)
    return [attention_context, residual]

def get_init_inputs():
    return [dim]
