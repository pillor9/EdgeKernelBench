# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: vit_patch_embedding.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_161_vit_patch_embedding',
 'level': 2,
 'name': 'vit_patch_embedding',
 'scenario_tags': ['embodied_robotics'],
 'operator_tags': ['convolution', 'matrix_multiplication', 'normalization', 'embedding', 'tensor_transform'],
 'semantic_references': [{'url': 'https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py',
                          'symbol': 'PatchEmbed'}],
 'summary': 'timm PatchEmbed: Conv2d patch projection -> flatten(2) -> transpose(1,2) -> LayerNorm'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
dim = __EKB_PARAM_dim__
patch_size = __EKB_PARAM_patch_size__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, dim, patch_size]
