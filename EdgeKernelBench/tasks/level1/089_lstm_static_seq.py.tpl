# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: lstm_static_seq.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l1_089_lstm_static_seq',
 'level': 1,
 'name': 'lstm_static_seq',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['recurrent'],
 'semantic_references': [{'url': 'https://github.com/zeezou-isee/MobileKernelBench_git/blob/main/dataset/Mobilekernelbench/Others/LSTM.py',
                          'symbol': 'nn.LSTM'}],
 'summary': 'Static single-layer unidirectional LSTM with sequence and hidden dimensions supplied by the '
            'profile.'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (h_n, c_n) = self.lstm(x)
        return output
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
input_size = __EKB_PARAM_input_size__
hidden_size = __EKB_PARAM_hidden_size__

def get_inputs():
    return [torch.rand(batch_size, seq_len, input_size)]

def get_init_inputs():
    return [input_size, hidden_size]
