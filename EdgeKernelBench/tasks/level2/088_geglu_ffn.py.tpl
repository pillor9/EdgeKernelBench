# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: geglu_ffn.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_088_geglu_ffn',
 'level': 2,
 'name': 'geglu_ffn',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['attention', 'matrix_multiplication', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py'}],
 'summary': 'Source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py '
            '(GEGLU/FeedForward).',
 'adaptation_notes': ['GLU is explicitly decomposed as linear, chunk, GELU, multiply, and output linear.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2 * hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = torch.chunk(self.proj(x), 2, dim=-1)
        return self.out(value * F.gelu(gate))
batch_size = __EKB_PARAM_batch_size__
tokens = __EKB_PARAM_tokens__
dim = __EKB_PARAM_dim__
hidden_dim = __EKB_PARAM_hidden_dim__

def get_inputs():
    return [torch.randn(batch_size, tokens, dim)]

def get_init_inputs():
    return [dim, hidden_dim]
