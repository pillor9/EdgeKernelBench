# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: dense_four_expert_moe.

Workload parameters are supplied through a scenario profile."""
from __future__ import annotations
TASK_METADATA = {'task_id': 'ekb_l2_066_dense_four_expert_moe',
 'level': 2,
 'name': 'dense_four_expert_moe',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['moe',
                   'matrix_multiplication',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlc-ai/mlc-llm/blob/2008fe8343e1f40ef89ee57b9287aebcf1b86c98/python/mlc_llm/model/mixtral/mixtral_model.py',
                          'symbol': 'MixtralMoE'}],
 'summary': 'Dense four-expert SwiGLU MoE surrogate with softmax routing over every expert.',
 'adaptation_notes': ['Top-k, token scatter/gather, expert parallelism, and packed kernels are deliberately '
                      'replaced by evaluating all four experts and taking a weighted sum.'],
 'model_families': ['Mixtral']}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUExpert(nn.Module):

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Model(nn.Module):

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.router = nn.Linear(dim, 4, bias=False)
        self.expert0 = SwiGLUExpert(dim, ffn_dim)
        self.expert1 = SwiGLUExpert(dim, ffn_dim)
        self.expert2 = SwiGLUExpert(dim, ffn_dim)
        self.expert3 = SwiGLUExpert(dim, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routing = torch.softmax(self.router(x), dim=-1)
        expert_outputs = torch.stack((self.expert0(x), self.expert1(x), self.expert2(x), self.expert3(x)), dim=-2)
        return torch.sum(expert_outputs * routing.unsqueeze(-1), dim=-2)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__
ffn_dim = __EKB_PARAM_ffn_dim__

def get_inputs():
    torch.manual_seed(__EKB_INPUT_get_inputs_001_c001__)
    return [torch.randn(batch_size, seq_len, dim, dtype=torch.float32)]

def get_init_inputs():
    return [dim, ffn_dim]
