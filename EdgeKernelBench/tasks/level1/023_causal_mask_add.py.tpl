# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: causal_mask_add.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l1_023_causal_mask_add',
 'level': 1,
 'name': 'causal_mask_add',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/qwen2/qwen2_model.py',
                          'symbol': 'QWen2Attention'}],
 'summary': 'Add a broadcast causal mask to prefill attention scores.',
 'adaptation_notes': ['Pass a fixed fp32 additive mask explicitly instead of constructing runtime cache '
                      'masks.'],
 'model_families': ['Qwen2']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def forward(self, scores: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        return scores + causal_mask
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
num_heads = __EKB_PARAM_num_heads__

def get_inputs():
    scores = torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=torch.float32)
    mask = torch.full((__EKB_INPUT_get_inputs_001_c001__, __EKB_INPUT_get_inputs_001_c002__, seq_len, seq_len), -__EKB_INPUT_get_inputs_001_c003__)
    mask = torch.triu(mask, diagonal=__EKB_INPUT_get_inputs_001_c004__ + (seq_len - seq_len))
    return [scores, mask]

def get_init_inputs():
    return []
