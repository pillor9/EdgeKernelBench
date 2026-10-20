# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: lm_head.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l1_084_lm_head',
 'level': 1,
 'name': 'lm_head',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['matrix_multiplication', 'embedding'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/llama/llama_model.py',
                          'symbol': 'LlamaForCausalLM.get_logits'}],
 'summary': 'Project prefill hidden states to full vocabulary logits.',
 'adaptation_notes': ['Use a bias-free fp32 torch.nn.Linear and omit distributed vocabulary sharding.'],
 'model_families': ['Llama']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
hidden_size = __EKB_PARAM_hidden_size__
vocab_size = __EKB_PARAM_vocab_size__

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)]

def get_init_inputs():
    return [hidden_size, vocab_size]
