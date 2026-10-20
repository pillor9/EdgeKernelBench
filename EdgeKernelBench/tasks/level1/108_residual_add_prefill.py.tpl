# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: residual_add_prefill.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l1_108_residual_add_prefill',
 'level': 1,
 'name': 'residual_add_prefill',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['residual', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/qwen2/qwen2_model.py',
                          'symbol': 'QWen2DecoderLayer'}],
 'summary': 'Merge a complete prefill sublayer result with the residual stream.',
 'adaptation_notes': ['Extract a broadcast-free fp32 tensor addition.'],
 'model_families': ['Qwen2']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def forward(self, residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return residual + update
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
hidden_size = __EKB_PARAM_hidden_size__

def get_inputs():
    shape = (batch_size, seq_len, hidden_size)
    return [torch.randn(shape, dtype=torch.float32), torch.randn(shape, dtype=torch.float32)]

def get_init_inputs():
    return []
