# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: logits_head_rmsnorm_linear_vocab.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_107_logits_head_rmsnorm_linear_vocab',
 'level': 2,
 'name': 'logits_head_rmsnorm_linear_vocab',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['matrix_multiplication', 'normalization', 'reduction', 'embedding', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py',
                          'symbol': 'LlamaRMSNorm / lm_head'}],
 'summary': 'final RMSNorm followed by vocabulary Linear projection'}
import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + self.eps))

class Model(nn.Module):

    def __init__(self, dim, vocab_size):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden_states))
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
dim = __EKB_PARAM_dim__
vocab_size = __EKB_PARAM_vocab_size__

def get_inputs():
    return [torch.rand(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, vocab_size]
