# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: gelu_prefill.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l1_060_gelu_prefill',
 'level': 1,
 'name': 'gelu_prefill',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['activation'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/qwen2/qwen2_model.py',
                          'symbol': 'QWen2MLP.act_fn'}],
 'summary': 'Apply an exact GELU control workload to a prefill FFN-width tensor.',
 'adaptation_notes': ['Use fp32 torch.nn.functional.gelu as an activation comparison to Qwen2 SiLU.'],
 'model_families': ['Qwen2 activation control']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(hidden_states, approximate='none')
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
intermediate_size = __EKB_PARAM_intermediate_size__

def get_inputs():
    return [torch.randn(batch_size, seq_len, intermediate_size, dtype=torch.float32)]

def get_init_inputs():
    return []
