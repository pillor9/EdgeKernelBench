# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: embedding.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_056_embedding',
 'level': 1,
 'name': 'embedding',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['embedding'],
 'semantic_references': [{'url': 'https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py',
                          'symbol': 'LlamaModel.embed_tokens'}],
 'summary': 'LlamaModel creates self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, '
            'padding_idx)'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
vocab_size = __EKB_PARAM_vocab_size__
hidden_size = __EKB_PARAM_hidden_size__

def get_inputs():
    return [torch.randint(__EKB_INPUT_get_inputs_001_c001__, vocab_size, (batch_size, seq_len), dtype=torch.long)]

def get_init_inputs():
    return [vocab_size, hidden_size]
