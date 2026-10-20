# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: postnorm_residual.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_121_postnorm_residual',
 'level': 2,
 'name': 'postnorm_residual',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['matrix_multiplication', 'normalization', 'activation', 'residual', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py',
                          'symbol': 'ResPostBlock'}],
 'summary': 'timm ResPostBlock MLP half: x + LayerNorm(Mlp(x)); dropout is eval identity'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm(x)
        return residual + x
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__
hidden_dim = __EKB_PARAM_hidden_dim__

def get_inputs():
    return [torch.rand(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, hidden_dim]
