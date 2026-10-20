# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: decode_linear_gemv.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l1_038_decode_linear_gemv',
 'level': 1,
 'name': 'decode_linear_gemv',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['matrix_multiplication'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/llama/llama_model.py',
                          'symbol': 'LlamaAttention.qkv_proj'}],
 'summary': 'Project one decode-token hidden state with a bias-free dense layer (GEMV regime).',
 'adaptation_notes': ['Extract one Llama projection as a static fp32 torch.nn.Linear primitive.'],
 'model_families': ['Llama']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
hidden_size = __EKB_PARAM_hidden_size__

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)]

def get_init_inputs():
    return [hidden_size]
