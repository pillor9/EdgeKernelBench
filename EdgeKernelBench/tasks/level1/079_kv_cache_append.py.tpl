# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: kv_cache_append.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l1_079_kv_cache_append',
 'level': 1,
 'name': 'kv_cache_append',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['tensor_transform'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/llama/llama_model.py',
                          'symbol': 'LlamaForCausalLM.batch_decode'}],
 'summary': "Append one token's key vectors to an explicit static decode cache.",
 'adaptation_notes': ['Replace mutation-oriented PagedKVCache state with fp32 tensors and torch.cat.'],
 'model_families': ['Llama']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def forward(self, key_cache: torch.Tensor, new_key: torch.Tensor) -> torch.Tensor:
        return torch.cat((key_cache, new_key), dim=2)
batch_size = __EKB_PARAM_batch_size__
num_kv_heads = __EKB_PARAM_num_kv_heads__
cache_len = __EKB_PARAM_cache_len__
seq_len = __EKB_PARAM_seq_len__
head_dim = __EKB_PARAM_head_dim__

def get_inputs():
    key_cache = torch.randn(batch_size, num_kv_heads, cache_len, head_dim, dtype=torch.float32)
    new_key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.float32)
    return [key_cache, new_key]

def get_init_inputs():
    return []
