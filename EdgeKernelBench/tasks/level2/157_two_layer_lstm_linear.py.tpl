# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: two_layer_lstm_linear.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_157_two_layer_lstm_linear',
 'level': 2,
 'name': 'two_layer_lstm_linear',
 'scenario_tags': ['smart_aiot'],
 'operator_tags': ['matrix_multiplication', 'recurrent'],
 'semantic_references': [],
 'summary': 'two-layer LSTM over static sensor windows followed by a Linear regression head on the final '
            'timestep'}
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x, (h0, c0))
        return self.head(y[:, 47, :])
batch_size = __EKB_PARAM_batch_size__
seq_len = __EKB_PARAM_seq_len__
input_dim = __EKB_PARAM_input_dim__
hidden_dim = __EKB_PARAM_hidden_dim__
out_dim = __EKB_PARAM_out_dim__

def get_inputs():
    x = torch.rand(batch_size, seq_len, input_dim)
    h0 = torch.rand(__EKB_INPUT_get_inputs_001_c001__, batch_size, hidden_dim)
    c0 = torch.rand(__EKB_INPUT_get_inputs_001_c002__, batch_size, hidden_dim)
    return [x, h0, c0]

def get_init_inputs():
    return [input_dim, hidden_dim, out_dim]
