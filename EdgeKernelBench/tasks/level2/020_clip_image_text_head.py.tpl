# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: clip_image_text_head.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_020_clip_image_text_head',
 'level': 2,
 'name': 'clip_image_text_head',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'image projection branch and text projection branch are normalized and joined by scaled matrix '
            'product'}
import math
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, dim, out_dim):
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(channels, out_dim, 3, stride=2, padding=1), nn.ReLU())
        self.text = nn.Linear(dim, out_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-06)

    def forward(self, image: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        image_feat = self.normalize(self.image(image).mean(dim=(-2, -1)))
        text_feat = self.normalize(self.text(text_tokens))
        return self.logit_scale.exp() * torch.matmul(image_feat, text_feat.transpose(0, 1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
dim = __EKB_PARAM_dim__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
seq_len = __EKB_PARAM_seq_len__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width), torch.rand(seq_len, dim)]

def get_init_inputs():
    return [channels, dim, out_dim]
