# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Modified for EdgeKernelBench: task extraction and scenario parameterization.
# Licensing notices updated 2026-10-06.
# Upstream license: LICENSES/huggingface--transformers/LICENSE.txt
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: rmsnorm_decomposed.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_111_rmsnorm_decomposed',
 'level': 1,
 'name': 'rmsnorm_decomposed',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['normalization', 'reduction', 'type_conversion', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py',
                          'symbol': 'LlamaRMSNorm'}],
 'summary': 'hidden_states.to(fp32); variance=pow(2).mean(-1,keepdim=True); hidden*rsqrt(variance+eps); '
            'weight multiply'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, hidden_size: int, eps: float=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
hidden_size = __EKB_PARAM_hidden_size__

def get_inputs():
    return [torch.rand(batch_size, seq_len, hidden_size)]

def get_init_inputs():
    return [hidden_size, __EKB_INPUT_get_init_inputs_001_c001__]
