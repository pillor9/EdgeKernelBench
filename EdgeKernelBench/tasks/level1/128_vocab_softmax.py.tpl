# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: vocab_softmax.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_128_vocab_softmax',
 'level': 1,
 'name': 'vocab_softmax',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['activation', 'embedding', 'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py',
                          'symbol': 'LlamaForCausalLM logits'}],
 'summary': 'softmax over vocabulary logits after the Llama lm_head'}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1)
batch_size = __EKB_PARAM_batch_size__
vocab_size = __EKB_PARAM_vocab_size__

def get_inputs():
    return [torch.rand(batch_size, vocab_size)]

def get_init_inputs():
    return []
