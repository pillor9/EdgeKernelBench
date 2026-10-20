# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: efficientformer_poolmixer.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_077_efficientformer_poolmixer',
 'level': 2,
 'name': 'efficientformer_poolmixer',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['convolution', 'activation', 'pooling', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/snap-research/EfficientFormer'}],
 'summary': 'Source: https://github.com/snap-research/EfficientFormer (Pooling/Meta4D).',
 'adaptation_notes': ['PoolMixer token path avgpool(x)-x with residual add, followed by pointwise MLP '
                      'residual.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, c: int, hidden: int):
        super().__init__()
        self.a = nn.Conv2d(c, hidden, 1)
        self.b = nn.Conv2d(hidden, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + (F.avg_pool2d(x, 3, stride=1, padding=1) - x)
        return x + self.b(F.gelu(self.a(x)))
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
hidden = __EKB_PARAM_hidden__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, hidden]
