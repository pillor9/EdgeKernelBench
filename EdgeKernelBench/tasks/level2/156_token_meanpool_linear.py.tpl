# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: token_meanpool_linear.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_156_token_meanpool_linear',
 'level': 2,
 'name': 'token_meanpool_linear',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['matrix_multiplication', 'pooling', 'reduction'],
 'semantic_references': [],
 'summary': 'static token mean-pool followed by Linear projection'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = tokens.mean(dim=1)
        return self.proj(pooled)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__
out_dim = __EKB_PARAM_out_dim__

def get_inputs():
    return [torch.rand(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, out_dim]
