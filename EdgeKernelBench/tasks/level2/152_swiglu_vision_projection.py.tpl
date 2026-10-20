# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: swiglu_vision_projection.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_152_swiglu_vision_projection',
 'level': 2,
 'name': 'swiglu_vision_projection',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'SwiGLU token FFN branch and vision Conv projection branch are independent until concat+Linear '
            'fusion'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, dim, hidden_dim, out_dim):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, out_dim, bias=False)
        self.vision = nn.Sequential(nn.Conv2d(channels, out_dim, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim), nn.SiLU())
        self.fuse = nn.Linear(out_dim * 2, out_dim)

    def forward(self, image: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        token_feat = self.down(torch.sigmoid(self.gate(tokens)) * self.gate(tokens) * self.up(tokens)).mean(dim=1)
        image_feat = self.vision(image).mean(dim=(-2, -1))
        return self.fuse(torch.cat((token_feat, image_feat), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
dim = __EKB_PARAM_dim__
hidden_dim = __EKB_PARAM_hidden_dim__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
seq_len = __EKB_PARAM_seq_len__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width), torch.rand(batch_size, seq_len, dim)]

def get_init_inputs():
    return [channels, dim, hidden_dim, out_dim]
