# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: mlp_mixer_block.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_113_mlp_mixer_block',
 'level': 2,
 'name': 'mlp_mixer_block',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['matrix_multiplication', 'normalization', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mlp_mixer.py'}],
 'summary': 'Source: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mlp_mixer.py '
            '(MixerBlock).',
 'adaptation_notes': ['pre-norm token-mixing MLP followed by residual and pre-norm channel-mixing MLP '
                      'followed by residual.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, tokens: int, dim: int, token_hidden: int, channel_hidden: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token1 = nn.Linear(tokens, token_hidden)
        self.token2 = nn.Linear(token_hidden, tokens)
        self.norm2 = nn.LayerNorm(dim)
        self.channel1 = nn.Linear(dim, channel_hidden)
        self.channel2 = nn.Linear(channel_hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x).transpose(1, 2)
        x = x + self.token2(F.gelu(self.token1(y))).transpose(1, 2)
        y = self.norm2(x)
        return x + self.channel2(F.gelu(self.channel1(y)))
batch_size = __EKB_PARAM_batch_size__
tokens = __EKB_PARAM_tokens__
dim = __EKB_PARAM_dim__
token_hidden = __EKB_PARAM_token_hidden__
channel_hidden = __EKB_PARAM_channel_hidden__

def get_inputs():
    return [torch.randn(batch_size, tokens, dim)]

def get_init_inputs():
    return [tokens, dim, token_hidden, channel_hidden]
