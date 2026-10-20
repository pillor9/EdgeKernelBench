# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: mlp_mixer_tiny.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_019_mlp_mixer_tiny',
 'level': 3,
 'name': 'mlp_mixer_tiny',
 'scenario_tags': ['smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mlp_mixer.py'}],
 'summary': 'Source: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mlp_mixer.py '
            '(MlpMixer).',
 'adaptation_notes': ['patch embedding, two pre-norm token/channel Mixer blocks, LayerNorm, token average, '
                      'and classifier.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixerBlock(nn.Module):

    def __init__(self, tokens: int, dim: int, token_hidden: int, channel_hidden: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.t1 = nn.Linear(tokens, token_hidden)
        self.t2 = nn.Linear(token_hidden, tokens)
        self.norm2 = nn.LayerNorm(dim)
        self.c1 = nn.Linear(dim, channel_hidden)
        self.c2 = nn.Linear(channel_hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.t2(F.gelu(self.t1(self.norm1(x).transpose(1, 2)))).transpose(1, 2)
        return x + self.c2(F.gelu(self.c1(self.norm2(x))))

class Model(nn.Module):

    def __init__(self, image_size: int, patch_size: int, dim: int, token_hidden: int, channel_hidden: int, num_classes: int):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        tokens = (image_size // patch_size) ** 2
        self.blocks = nn.Sequential(MixerBlock(tokens, dim, token_hidden, channel_hidden), MixerBlock(tokens, dim, token_hidden, channel_hidden))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).flatten(2).transpose(1, 2)
        return self.head(self.norm(self.blocks(x)).mean(dim=1))
batch_size = __EKB_PARAM_batch_size__
image_size = __EKB_PARAM_image_size__
patch_size = __EKB_PARAM_patch_size__
dim = __EKB_PARAM_dim__
token_hidden = __EKB_PARAM_token_hidden__
channel_hidden = __EKB_PARAM_channel_hidden__
num_classes = __EKB_PARAM_num_classes__

def get_inputs():
    return [torch.randn(batch_size, __EKB_INPUT_get_inputs_001_c001__, image_size, image_size)]

def get_init_inputs():
    return [image_size, patch_size, dim, token_hidden, channel_hidden, num_classes]
