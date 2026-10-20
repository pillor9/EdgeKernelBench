# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: silu_prefill.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l1_118_silu_prefill',
 'level': 1,
 'name': 'silu_prefill',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['matrix_multiplication', 'activation'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/llama/llama_model.py',
                          'symbol': 'LlamaFFN'}],
 'summary': 'Apply SiLU to the prefill FFN gate projection.',
 'adaptation_notes': ['Isolate the fp32 torch.nn.functional.silu elementwise primitive.'],
 'model_families': ['Llama']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        return F.silu(gate)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
intermediate_size = __EKB_PARAM_intermediate_size__

def get_inputs():
    return [torch.randn(batch_size, seq_len, intermediate_size, dtype=torch.float32)]

def get_init_inputs():
    return []
