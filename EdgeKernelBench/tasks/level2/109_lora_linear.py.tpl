# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: lora_linear.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_109_lora_linear',
 'level': 2,
 'name': 'lora_linear',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['matrix_multiplication', 'low_rank_adaptation', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/microsoft/LoRA/blob/main/loralib/layers.py'}],
 'summary': 'Source: https://github.com/microsoft/LoRA/blob/main/loralib/layers.py (Linear).',
 'adaptation_notes': ['Fixed-shape linear projection with a scaled low-rank residual: base(x) + scale * B(A(x)).']}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, dim: int, out: int, rank: int, scale: float):
        super().__init__()
        self.base = nn.Linear(dim, out)
        self.a = nn.Linear(dim, rank, bias=False)
        self.b = nn.Linear(rank, out, bias=False)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.b(self.a(x))
batch_size = __EKB_PARAM_batch_size__
tokens = __EKB_PARAM_tokens__
dim = __EKB_PARAM_dim__
out = __EKB_PARAM_out__
rank = __EKB_PARAM_rank__
scale = __EKB_PARAM_scale__

def get_inputs():
    return [torch.randn(batch_size, tokens, dim)]

def get_init_inputs():
    return [dim, out, rank, scale]
