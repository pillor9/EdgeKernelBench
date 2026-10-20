# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/huggingface--pytorch-image-models/LICENSE.txt
# Upstream license: LICENSES/microsoft--Swin-Transformer/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: swin_patch_merge.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_153_swin_patch_merge',
 'level': 2,
 'name': 'swin_patch_merge',
 'scenario_tags': ['mobile_personal', 'smart_aiot'],
 'operator_tags': ['matrix_multiplication', 'normalization', 'reduction', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer.py'}],
 'summary': 'Source: '
            'https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer.py '
            '(PatchMerging).',
 'adaptation_notes': ['four static strided spatial slices, channel concat, LayerNorm, and linear reduction.']}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        return self.reduction(self.norm(torch.cat([x0, x1, x2, x3], dim=-1)))
batch_size = __EKB_PARAM_batch_size__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
dim = __EKB_PARAM_dim__

def get_inputs():
    return [torch.randn(batch_size, height, width, dim)]

def get_init_inputs():
    return [dim]
