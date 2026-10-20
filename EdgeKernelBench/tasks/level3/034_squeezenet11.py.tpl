# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: squeezenet11.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_034_squeezenet11',
 'level': 3,
 'name': 'squeezenet11',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution', 'activation', 'tensor_transform'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py'}],
 'summary': 'Source: https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py '
            '(SqueezeNet1_1).',
 'adaptation_notes': ['Compact SqueezeNet1.1-inspired classifier with a stem, four Fire modules, '
                      'pooling, a final convolution, global average, and logits.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Fire(nn.Module):

    def __init__(self, ci: int, sq: int, ex: int):
        super().__init__()
        self.s = nn.Conv2d(ci, sq, 1)
        self.e1 = nn.Conv2d(sq, ex, 1)
        self.e3 = nn.Conv2d(sq, ex, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.s(x))
        return torch.cat([F.relu(self.e1(x)), F.relu(self.e3(x))], 1)

class Model(nn.Module):

    def __init__(self, stem: int, squeeze: int, expand: int, classes: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, stride=2), nn.ReLU(), nn.MaxPool2d(3, stride=2, ceil_mode=True))
        self.features = nn.Sequential(Fire(stem, squeeze, expand), Fire(2 * expand, squeeze, expand), nn.MaxPool2d(3, stride=2, ceil_mode=True), Fire(2 * expand, 2 * squeeze, 2 * expand), Fire(4 * expand, 2 * squeeze, 2 * expand))
        self.final = nn.Conv2d(4 * expand, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(F.relu(self.final(self.features(self.stem(x)))), 1).flatten(1)
batch_size = __EKB_PARAM_batch_size__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
stem = __EKB_PARAM_stem__
squeeze = __EKB_PARAM_squeeze__
expand = __EKB_PARAM_expand__
classes = __EKB_PARAM_classes__

def get_inputs():
    return [torch.randn(batch_size, __EKB_INPUT_get_inputs_001_c001__, height, width)]

def get_init_inputs():
    return [stem, squeeze, expand, classes]
