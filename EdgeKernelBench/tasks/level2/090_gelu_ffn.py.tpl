# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: gelu_ffn.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l2_090_gelu_ffn',
 'level': 2,
 'name': 'gelu_ffn',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['matrix_multiplication', 'activation'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/gpt2/gpt2_model.py',
                          'symbol': 'GPT2MLP'}],
 'summary': 'Prefill two-layer GELU feed-forward sublayer.',
 'adaptation_notes': ['GELU replaces the referenced configurable gated activation to cover a common LLM FFN '
                      'variant while retaining static projection ratios.'],
 'model_families': ['Transformer GELU']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__
ffn_dim = __EKB_PARAM_ffn_dim__

def get_inputs():
    torch.manual_seed(__EKB_INPUT_get_inputs_001_c001__)
    return [torch.randn(batch_size, seq_len, dim, dtype=torch.float32)]

def get_init_inputs():
    return [dim, ffn_dim]
