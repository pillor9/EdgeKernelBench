# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: gemv_decode.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_067_gemv_decode',
 'level': 1,
 'name': 'gemv_decode',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['matrix_multiplication'],
 'semantic_references': [{'url': 'https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py',
                          'symbol': 'LlamaMLP projection'}],
 'summary': 'Bias-free dense projection of a decode-token tensor, following the LlamaMLP projection pattern.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
batch_size = __EKB_PARAM_batch_size__
in_features = __EKB_PARAM_in_features__
out_features = __EKB_PARAM_out_features__

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
