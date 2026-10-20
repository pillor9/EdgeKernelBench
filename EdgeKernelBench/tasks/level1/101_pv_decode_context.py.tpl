# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: pv_decode_context.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l1_101_pv_decode_context',
 'level': 1,
 'name': 'pv_decode_context',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['attention', 'matrix_multiplication', 'activation'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/llama/llama_model.py',
                          'symbol': 'LlamaAttention'}],
 'summary': 'Multiply decode attention probabilities by explicit cached values.',
 'adaptation_notes': ['Replace paged-cache value aggregation with a static fp32 torch.matmul.'],
 'model_families': ['Llama']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def forward(self, attention_probs: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.matmul(attention_probs, value)
batch_size = __EKB_PARAM_batch_size__
num_heads = __EKB_PARAM_num_heads__
query_len = __EKB_PARAM_query_len__
key_len = __EKB_PARAM_key_len__
head_dim = __EKB_PARAM_head_dim__

def get_inputs():
    logits = torch.randn(batch_size, num_heads, query_len, key_len, dtype=torch.float32)
    attention_probs = F.softmax(logits, dim=-__EKB_INPUT_get_inputs_001_c001__)
    value = torch.randn(batch_size, num_heads, key_len, head_dim, dtype=torch.float32)
    return [attention_probs, value]

def get_init_inputs():
    return []
