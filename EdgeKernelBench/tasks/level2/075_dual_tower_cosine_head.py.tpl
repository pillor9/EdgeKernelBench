# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: dual_tower_cosine_head.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_075_dual_tower_cosine_head',
 'level': 2,
 'name': 'dual_tower_cosine_head',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['matrix_multiplication', 'reduction', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py',
                          'symbol': 'CLIP.forward'}],
 'summary': 'CLIP dual-tower head: L2 normalize image/text features then scaled matrix multiply logits'}
import math
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-06)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        image_features = self._normalize(image_features)
        text_features = self._normalize(text_features)
        return self.logit_scale.exp() * torch.matmul(image_features, text_features.transpose(0, 1))
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__

def get_inputs():
    image_features = torch.rand(batch_size, dim)
    text_features = torch.rand(seq_len, dim)
    return [image_features, text_features]

def get_init_inputs():
    return []
