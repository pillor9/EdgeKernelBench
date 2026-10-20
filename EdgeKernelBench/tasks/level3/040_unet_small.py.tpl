# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: unet_small.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l3_040_unet_small',
 'level': 3,
 'name': 'unet_small',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['convolution',
                   'normalization',
                   'activation',
                   'resampling',
                   'tensor_transform',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py'}],
 'summary': 'Source: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py (UNet).',
 'adaptation_notes': ['static small U-Net with two encoder scales, bottleneck, bilinear upsampling, skip '
                      'concatenation, decoder convolutions and segmentation head.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Double(nn.Module):

    def __init__(self, ci: int, co: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(), nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Model(nn.Module):

    def __init__(self, c1: int, c2: int, c3: int, classes: int):
        super().__init__()
        self.e1 = Double(3, c1)
        self.e2 = Double(c1, c2)
        self.mid = Double(c2, c3)
        self.d2 = Double(c3 + c2, c2)
        self.d1 = Double(c2 + c1, c1)
        self.head = nn.Conv2d(c1, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.e1(x)
        s2 = self.e2(F.max_pool2d(s1, 2))
        y = self.mid(F.max_pool2d(s2, 2))
        y = F.interpolate(y, size=s2.shape[-2:], mode='bilinear', align_corners=False)
        y = self.d2(torch.cat([y, s2], 1))
        y = F.interpolate(y, size=s1.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(self.d1(torch.cat([y, s1], 1)))
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
