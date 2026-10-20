# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: ghostnet_embedding_variant.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_092_ghostnet_embedding_variant',
 'level': 2,
 'name': 'ghostnet_embedding_variant',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'embedding',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'GhostNet-style vision branch and token projection branch are independent until concat+Linear '
            'fusion'}
import torch
import torch.nn as nn

class GhostModule(nn.Module):

    def __init__(self, channels, out_channels):
        super().__init__()
        primary = out_channels // 2
        cheap = out_channels - primary
        self.primary = nn.Sequential(nn.Conv2d(channels, primary, 1, bias=False), nn.BatchNorm2d(primary), nn.ReLU())
        self.cheap = nn.Sequential(nn.Conv2d(primary, cheap, 3, padding=1, groups=primary, bias=False), nn.BatchNorm2d(cheap), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.primary(x)
        return torch.cat((y, self.cheap(y)), dim=1)

class Model(nn.Module):

    def __init__(self, channels, dim, out_channels, out_dim):
        super().__init__()
        self.ghost = GhostModule(channels, out_channels)
        self.vision_proj = nn.Linear(out_channels, out_dim)
        self.text = nn.Sequential(nn.Linear(dim, out_dim), nn.GELU(), nn.LayerNorm(out_dim))
        self.fuse = nn.Linear(out_dim * 2, out_dim)

    def forward(self, image: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        image_feat = self.vision_proj(self.ghost(image).mean(dim=(-2, -1)))
        token_feat = self.text(tokens).mean(dim=1)
        return self.fuse(torch.cat((image_feat, token_feat), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
dim = __EKB_PARAM_dim__
out_channels = __EKB_PARAM_out_channels__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
seq_len = __EKB_PARAM_seq_len__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width), torch.rand(batch_size, seq_len, dim)]

def get_init_inputs():
    return [channels, dim, out_channels, out_dim]
