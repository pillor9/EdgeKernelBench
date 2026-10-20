# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: globalavgpool_linear_sigmoid_head.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_093_globalavgpool_linear_sigmoid_head',
 'level': 2,
 'name': 'globalavgpool_linear_sigmoid_head',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['matrix_multiplication', 'activation', 'pooling', 'reduction', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py',
                          'symbol': 'classification_head'}],
 'summary': 'classification head variant: global average pool -> flatten -> Linear -> Sigmoid'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, num_classes):
        super().__init__()
        self.fc = nn.Linear(channels, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=(-2, -1))
        x = self.fc(x)
        return self.sigmoid(x)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
num_classes = __EKB_PARAM_num_classes__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, num_classes]
