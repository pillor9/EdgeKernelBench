# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: attention_softmax_prefill.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l1_015_attention_softmax_prefill',
 'level': 1,
 'name': 'attention_softmax_prefill',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention', 'activation', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/llama/llama_model.py',
                          'symbol': 'LlamaAttention'}],
 'summary': 'Normalize every prefill attention row across the key sequence.',
 'adaptation_notes': ['Isolate the fp32 last-axis torch.softmax from fused attention.'],
 'model_families': ['Llama']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def forward(self, attention_logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(attention_logits, dim=-1)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
num_heads = __EKB_PARAM_num_heads__

def get_inputs():
    return [torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=torch.float32)]

def get_init_inputs():
    return []
