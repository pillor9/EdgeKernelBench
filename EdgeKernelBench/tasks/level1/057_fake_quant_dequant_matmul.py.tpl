# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: fake_quant_dequant_matmul.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_057_fake_quant_dequant_matmul',
 'level': 1,
 'name': 'fake_quant_dequant_matmul',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['matrix_multiplication', 'quantization', 'type_conversion', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mit-han-lab/llm-awq', 'symbol': 'dequant_matmul'}],
 'summary': 'deployment dequant chain (w_int.float() - zero_point) * scale followed by matmul'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, w_int: torch.Tensor, zero_point: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        weight = (w_int.float() - zero_point) * scale
        return x @ weight
batch_size = __EKB_PARAM_batch_size__
in_features = __EKB_PARAM_in_features__
out_features = __EKB_PARAM_out_features__

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w_int = torch.randint(-__EKB_INPUT_get_inputs_001_c001__, __EKB_INPUT_get_inputs_001_c002__, (in_features, out_features), dtype=torch.int8)
    zero_point = torch.zeros(__EKB_INPUT_get_inputs_001_c003__, out_features)
    scale = torch.rand(__EKB_INPUT_get_inputs_001_c004__, out_features)
    return [x, w_int, zero_point, scale]

def get_init_inputs():
    return []
