# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: repeat_kv_expand.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l1_105_repeat_kv_expand',
 'level': 1,
 'name': 'repeat_kv_expand',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention', 'tensor_transform'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/qwen2/qwen2_model.py',
                          'symbol': 'QWen2Attention'}],
 'summary': 'Expand grouped-query key/value heads to the query-head count.',
 'adaptation_notes': ['Replace cache-specific head replication with static reshape-expand-reshape '
                      'operations.'],
 'model_families': ['Qwen2']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def repeat_kv(x: torch.Tensor, num_q_heads: int, num_kv_heads: int) -> torch.Tensor:
    groups = num_q_heads // num_kv_heads
    batch, _, tokens, head_dim = x.shape
    return x.reshape(batch, num_kv_heads, 1, tokens, head_dim).expand(batch, num_kv_heads, groups, tokens, head_dim).reshape(batch, num_q_heads, tokens, head_dim)

class Model(nn.Module):

    def __init__(self, num_q_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

    def forward(self, key: torch.Tensor) -> torch.Tensor:
        return repeat_kv(key, self.num_q_heads, self.num_kv_heads)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
num_q_heads = __EKB_PARAM_num_q_heads__
num_kv_heads = __EKB_PARAM_num_kv_heads__
head_dim = __EKB_PARAM_head_dim__

def get_inputs():
    return [torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.float32)]

def get_init_inputs():
    return [num_q_heads, num_kv_heads]
