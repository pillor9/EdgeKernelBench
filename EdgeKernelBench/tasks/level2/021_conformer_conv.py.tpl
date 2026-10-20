# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: conformer_conv.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_021_conformer_conv',
 'level': 2,
 'name': 'conformer_conv',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution', 'normalization', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/sooftware/conformer/blob/main/conformer/convolution.py'}],
 'summary': 'Source: https://github.com/sooftware/conformer/blob/main/conformer/convolution.py '
            '(ConformerConvModule).',
 'adaptation_notes': ['pointwise projection, GLU decomposed as chunk-sigmoid-multiply, depthwise Conv1d, '
                      'BatchNorm, SiLU and output projection.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.n = nn.LayerNorm(dim)
        self.a = nn.Conv1d(dim, 2 * dim, 1)
        self.d = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)
        self.b = nn.BatchNorm1d(dim)
        self.o = nn.Conv1d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.a(self.n(x).transpose(1, 2))
        a, b = torch.chunk(y, 2, dim=1)
        y = a * torch.sigmoid(b)
        return self.o(F.silu(self.b(self.d(y)))).transpose(1, 2)
batch_size = __EKB_PARAM_batch_size__
tokens = __EKB_PARAM_tokens__
dim = __EKB_PARAM_dim__

def get_inputs():
    return [torch.randn(batch_size, tokens, dim)]

def get_init_inputs():
    return [dim]
