# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: fire_module.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_082_fire_module',
 'level': 2,
 'name': 'fire_module',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution', 'activation', 'tensor_transform', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py',
                          'symbol': 'Fire'}],
 'summary': 'Fire applies squeeze 1x1 conv+ReLU, then parallel expand 1x1 and expand 3x3 conv+ReLU branches, '
            'then concat'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, channels, squeeze_channels, out_channels):
        super().__init__()
        expand1 = out_channels // 2
        expand3 = out_channels - expand1
        self.squeeze = nn.Conv2d(channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU()
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1, kernel_size=1)
        self.expand1x1_activation = nn.ReLU()
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat((self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))), 1)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
squeeze_channels = __EKB_PARAM_squeeze_channels__
out_channels = __EKB_PARAM_out_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__

def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]

def get_init_inputs():
    return [channels, squeeze_channels, out_channels]
