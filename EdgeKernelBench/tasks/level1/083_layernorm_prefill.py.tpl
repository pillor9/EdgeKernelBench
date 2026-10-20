# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: layernorm_prefill.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l1_083_layernorm_prefill',
 'level': 1,
 'name': 'layernorm_prefill',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['normalization'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/qwen2/qwen2_model.py',
                          'symbol': 'QWen2DecoderLayer.input_layernorm'}],
 'summary': 'Apply a LayerNorm control workload at a Qwen2 decoder normalization site.',
 'adaptation_notes': ['Intentionally use fp32 torch.nn.LayerNorm as a compiler comparison to upstream '
                      'RMSNorm.'],
 'model_families': ['Qwen2 normalization control']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-05)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
hidden_size = __EKB_PARAM_hidden_size__

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)]

def get_init_inputs():
    return [hidden_size]
