# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: mobilenet_text_embedding.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_114_mobilenet_text_embedding',
 'level': 2,
 'name': 'mobilenet_text_embedding',
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
 'summary': 'MobileNet-style depthwise separable image branch and token embedding projection branch fuse by '
            'concat+Linear'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, dim, mid_channels, out_dim):
        super().__init__()
        self.mobile = nn.Sequential(nn.Conv2d(channels, mid_channels, 1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU6(), nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU6(), nn.Conv2d(mid_channels, out_dim, 1, bias=False), nn.BatchNorm2d(out_dim))
        self.text = nn.Sequential(nn.Linear(dim, out_dim), nn.GELU(), nn.LayerNorm(out_dim))
        self.fuse = nn.Linear(out_dim * 2, out_dim)

    def forward(self, image: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        image_feat = self.mobile(image).mean(dim=(-2, -1))
        text_feat = self.text(tokens).mean(dim=1)
        return self.fuse(torch.cat((image_feat, text_feat), dim=1))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
dim = __EKB_PARAM_dim__
mid_channels = __EKB_PARAM_mid_channels__
out_dim = __EKB_PARAM_out_dim__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
seq_len = __EKB_PARAM_seq_len__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width), torch.rand(batch_size, seq_len, dim)]

def get_init_inputs():
    return [channels, dim, mid_channels, out_dim]
