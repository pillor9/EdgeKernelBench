# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: layerscale_residual.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_104_layerscale_residual',
 'level': 2,
 'name': 'layerscale_residual',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['matrix_multiplication', 'residual', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py'}],
 'summary': 'Source: '
            'https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py '
            '(LayerScale).',
 'adaptation_notes': ['channelwise learned gamma scales a residual branch before addition.']}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.branch = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-05))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.branch(x)
batch_size = __EKB_PARAM_batch_size__
tokens = __EKB_PARAM_tokens__
dim = __EKB_PARAM_dim__

def get_inputs():
    return [torch.randn(batch_size, tokens, dim)]

def get_init_inputs():
    return [dim]
