# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: simam.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_137_simam',
 'level': 2,
 'name': 'simam',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['activation', 'reduction', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/ZjjConan/SimAM/blob/master/networks/attentions/simam_module.py'},
                         {'url': 'https://proceedings.mlr.press/v139/yang21o.html',
                          'section': 'Equations (5)-(6), Figure 3'}],
 'summary': 'Parameter-free spatial energy attention following the SimAM equations.',
 'adaptation_notes': ['Computes spatial mean, squared deviation, normalized energy, sigmoid attention, '
                      'and elementwise rescaling.']}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, e_lambda: float):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        deviation = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        energy = deviation / (4.0 * (deviation.sum(dim=(2, 3), keepdim=True) / (h * w - 1) + self.e_lambda)) + 0.5
        return x * torch.sigmoid(energy)
batch_size = __EKB_PARAM_batch_size__
channels = __EKB_PARAM_channels__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
e_lambda = __EKB_PARAM_e_lambda__

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    return [e_lambda]
