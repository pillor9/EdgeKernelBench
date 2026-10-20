# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: efficientnet_lite0.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_007_efficientnet_lite0',
 'level': 3,
 'name': 'efficientnet_lite0',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution',
                   'matrix_multiplication',
                   'normalization',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/pytorch-image-models'}],
 'summary': 'Source: https://github.com/huggingface/pytorch-image-models (efficientnet_lite0).',
 'adaptation_notes': ['Compact EfficientNet-Lite-inspired classifier with a ReLU6 stem, five depthwise '
                      'MBConv blocks, residual connections, an expansion head, global pooling, and a '
                      'classifier.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class MB(nn.Module):

    def __init__(self, ci: int, ce: int, co: int, s: int):
        super().__init__()
        self.use = ci == co and s == 1
        self.e = nn.Conv2d(ci, ce, 1, bias=False)
        self.eb = nn.BatchNorm2d(ce)
        self.d = nn.Conv2d(ce, ce, 3, stride=s, padding=1, groups=ce, bias=False)
        self.db = nn.BatchNorm2d(ce)
        self.p = nn.Conv2d(ce, co, 1, bias=False)
        self.pb = nn.BatchNorm2d(co)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu6(self.eb(self.e(x)))
        y = F.relu6(self.db(self.d(y)))
        y = self.pb(self.p(y))
        return x + y if self.use else y

class Model(nn.Module):

    def __init__(self, c1: int, c2: int, c3: int, classes: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, c1, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c1), nn.ReLU6())
        self.body = nn.Sequential(MB(c1, c1, c1, 1), MB(c1, 4 * c1, c2, 2), MB(c2, 4 * c2, c2, 1), MB(c2, 4 * c2, c3, 2), MB(c3, 4 * c3, c3, 1))
        self.last = nn.Conv2d(c3, 4 * c3, 1)
        self.head = nn.Linear(4 * c3, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu6(self.last(self.body(self.stem(x))))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))
batch_size = __EKB_PARAM_batch_size__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
c1 = __EKB_PARAM_c1__
c2 = __EKB_PARAM_c2__
c3 = __EKB_PARAM_c3__
classes = __EKB_PARAM_classes__

def get_inputs():
    return [torch.randn(batch_size, __EKB_INPUT_get_inputs_001_c001__, height, width)]

def get_init_inputs():
    return [c1, c2, c3, classes]
