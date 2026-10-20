# EdgeKernelBench licensing
# SPDX-License-Identifier: AGPL-3.0-only
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/ultralytics--ultralytics/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: dfl_expectation.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_071_dfl_expectation',
 'level': 2,
 'name': 'dfl_expectation',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['convolution', 'activation', 'tensor_transform', 'tensor_generation', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py',
                          'symbol': 'DFL'}],
 'summary': 'DFL reshapes box logits to four distributions, applies softmax over bins, and integrates with a '
            'fixed 1x1 conv'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, reg_max):
        super().__init__()
        self.reg_max = reg_max
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False).requires_grad_(False)
        bins = torch.arange(reg_max, dtype=torch.float32)
        self.conv.weight.data[:] = nn.Parameter(bins.view(1, reg_max, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(batch_size, 4, self.reg_max, num_anchors).transpose(2, 1)
        return self.conv(torch.softmax(x, dim=1)).view(batch_size, 4, num_anchors)
batch_size = __EKB_PARAM_batch_size__
reg_max = __EKB_PARAM_reg_max__
height = __EKB_PARAM_height__
width = __EKB_PARAM_width__
num_anchors = __EKB_PARAM_num_anchors__

def get_inputs():
    return [torch.rand(batch_size, __EKB_INPUT_get_inputs_001_c001__ * reg_max, num_anchors)]

def get_init_inputs():
    return [reg_max]
