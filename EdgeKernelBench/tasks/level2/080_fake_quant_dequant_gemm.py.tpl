# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: fake_quant_dequant_gemm.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_080_fake_quant_dequant_gemm',
 'level': 2,
 'name': 'fake_quant_dequant_gemm',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['matrix_multiplication',
                   'quantization',
                   'tensor_transform',
                   'type_conversion',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/mit-han-lab/llm-awq', 'symbol': 'dequant_gemm'}],
 'summary': 'AWQ-style weight-only path: cast quantized weight to fp32, apply per-output-channel scale, then '
            'GEMM'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, dim, out_dim):
        super().__init__()
        qweight = torch.randint(-8, 8, (out_dim, dim), dtype=torch.int8)
        scale = torch.rand(out_dim, 1) * 0.02
        self.register_buffer('qweight', qweight)
        self.register_buffer('scale', scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.qweight.to(torch.float32) * self.scale
        return torch.matmul(x, weight.transpose(0, 1))
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__
out_dim = __EKB_PARAM_out_dim__

def get_inputs():
    return [torch.rand(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, out_dim]
