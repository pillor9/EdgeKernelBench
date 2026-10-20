# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/zeezou-isee--MobileKernelBench_git/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: cast.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_022_cast',
 'level': 1,
 'name': 'cast',
 'scenario_tags': ['mobile_personal', 'smart_aiot', 'embodied_robotics'],
 'operator_tags': ['type_conversion'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Tensor/Cast.py',
                          'symbol': 'Cast'}],
 'summary': 'MobileKernelBench forward computation with workload parameters supplied by a scenario profile.'}
import torch
import torch.nn as nn
_DTYPE_MAP = {'float32': torch.float32, 'float16': torch.float16, 'float64': torch.float64, 'int32': torch.int32, 'int64': torch.int64, 'bool': torch.bool}

class Model(nn.Module):

    def __init__(self, to_dtype: str='int32'):
        super(Model, self).__init__()
        assert to_dtype in _DTYPE_MAP, f'Unsupported dtype: {to_dtype}'
        self.to_dtype_name = to_dtype
        self.to_dtype = _DTYPE_MAP[to_dtype]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.to_dtype)
batch_size = __EKB_PARAM_batch_size__
input_shape = __EKB_PARAM_input_shape__

def get_inputs():
    x = torch.randn(batch_size, *input_shape, dtype=torch.float32)
    return [x]

def get_init_inputs():
    return ['float16']
