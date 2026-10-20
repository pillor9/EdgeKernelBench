# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: lora_decode_linear.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l2_108_lora_decode_linear',
 'level': 2,
 'name': 'lora_decode_linear',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['matrix_multiplication', 'low_rank_adaptation', 'type_conversion', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/llama/llama_model.py',
                          'symbol': 'LlamaAttention.qkv_proj'}],
 'summary': 'Single-token dense projection plus a low-rank LoRA update.',
 'adaptation_notes': ['The low-rank branch is a portable benchmark extension around the referenced Llama '
                      'projection; weight loading and adapter runtime dispatch are excluded.'],
 'model_families': ['Llama-LoRA']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.base = nn.Linear(dim, dim, bias=False)
        self.lora_a = nn.Linear(dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, dim, bias=False)
        self.scale = 1.0 / float(rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        low_rank_output = self.lora_b(self.lora_a(x)) * self.scale
        return base_output + low_rank_output
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__
rank = __EKB_PARAM_rank__

def get_inputs():
    torch.manual_seed(__EKB_INPUT_get_inputs_001_c001__)
    return [torch.randn(batch_size, seq_len, dim, dtype=torch.float32)]

def get_init_inputs():
    return [dim, rank]
