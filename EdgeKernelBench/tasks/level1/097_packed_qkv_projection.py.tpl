# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: packed_qkv_projection.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l1_097_packed_qkv_projection',
 'level': 1,
 'name': 'packed_qkv_projection',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention', 'matrix_multiplication'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/qwen2/qwen2_model.py',
                          'symbol': 'QWen2Attention.qkv_proj'}],
 'summary': 'Produce a packed GQA query/key/value projection for a prefill sequence.',
 'adaptation_notes': ['Keep the packed fp32 dense output explicit and omit MLC tensor-parallel collectives.'],
 'model_families': ['Qwen2']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, hidden_size: int, packed_size: int):
        super().__init__()
        self.qkv_proj = nn.Linear(hidden_size, packed_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.qkv_proj(hidden_states)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
hidden_size = __EKB_PARAM_hidden_size__
num_q_heads = __EKB_PARAM_num_q_heads__
num_kv_heads = __EKB_PARAM_num_kv_heads__
head_dim = __EKB_PARAM_head_dim__
packed_size = __EKB_PARAM_packed_size__

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)]

def get_init_inputs():
    return [hidden_size, packed_size]
