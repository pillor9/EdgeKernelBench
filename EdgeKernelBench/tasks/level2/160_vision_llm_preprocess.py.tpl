# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: vision_llm_preprocess.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_160_vision_llm_preprocess',
 'level': 2,
 'name': 'vision_llm_preprocess',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'activation',
                   'reduction',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [],
 'summary': 'mobile vision Conv branch and LLM RMSNorm projection branch are independent until concat+Linear '
            'fusion'}
import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + self.eps))

class Model(nn.Module):

    def __init__(self, channels, dim, mid_channels, out_dim):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(channels, mid_channels, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(mid_channels), nn.Hardswish(), nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False), nn.BatchNorm2d(mid_channels), nn.Hardswish())
        self.norm = RMSNorm(dim)
        self.text_proj = nn.Linear(dim, out_dim, bias=False)
        self.fuse = nn.Linear(mid_channels + out_dim, out_dim)

    def forward(self, image: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        vision_feat = self.vision(image).mean(dim=(-2, -1))
        text_feat = self.text_proj(self.norm(tokens)).mean(dim=1)
        return self.fuse(torch.cat((vision_feat, text_feat), dim=1))
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
