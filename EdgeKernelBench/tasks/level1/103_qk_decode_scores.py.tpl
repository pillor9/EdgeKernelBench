# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: qk_decode_scores.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l1_103_qk_decode_scores',
 'level': 1,
 'name': 'qk_decode_scores',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['attention', 'matrix_multiplication', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/llama/llama_model.py',
                          'symbol': 'LlamaAttention'}],
 'summary': 'Compute scaled query-key scores for one query against an explicit decode cache.',
 'adaptation_notes': ['Replace paged-cache attention with an fp32 torch.matmul over explicit Q and K '
                      'tensors.'],
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
num_heads = __EKB_PARAM_num_heads__
query_len = __EKB_PARAM_query_len__
key_len = __EKB_PARAM_key_len__
head_dim = __EKB_PARAM_head_dim__

def get_inputs():
    query = torch.randn(batch_size, num_heads, query_len, head_dim, dtype=torch.float32)
    key = torch.randn(batch_size, num_heads, key_len, head_dim, dtype=torch.float32)
    return [query, key]

def get_init_inputs():
    return [head_dim]
