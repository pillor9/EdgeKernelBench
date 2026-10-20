# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: qk_prefill_scores.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l1_104_qk_prefill_scores',
 'level': 1,
 'name': 'qk_prefill_scores',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention', 'matrix_multiplication', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/llama/llama_model.py',
                          'symbol': 'LlamaAttention'}],
 'summary': 'Compute the scaled dense QK score matrix for a prefill sequence.',
 'adaptation_notes': ['Replace fused/paged attention with one explicit static fp32 torch.matmul.'],
 'model_families': ['Llama']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, head_dim: int):
        super().__init__()
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return torch.matmul(query, key.transpose(-1, -2)) * self.scale
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
num_heads = __EKB_PARAM_num_heads__
head_dim = __EKB_PARAM_head_dim__

def get_inputs():
    shape = (batch_size, num_heads, seq_len, head_dim)
    return [torch.randn(shape, dtype=torch.float32), torch.randn(shape, dtype=torch.float32)]

def get_init_inputs():
    return [head_dim]
