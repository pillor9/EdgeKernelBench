# EdgeKernelBench licensing
# SPDX-License-Identifier: Apache-2.0
# Scope and attribution: LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
# End licensing notice

"""Canonical EdgeKernelBench task: sliding_mask_gqa.

Workload parameters are supplied through a scenario profile."""
TASK_METADATA = {'task_id': 'ekb_l2_139_sliding_mask_gqa',
 'level': 2,
 'name': 'sliding_mask_gqa',
 'scenario_tags': ['mobile_personal'],
 'operator_tags': ['attention',
                   'matrix_multiplication',
                   'activation',
                   'tensor_transform',
                   'tensor_generation',
                   'elementwise'],
 'semantic_references': [{'url': 'https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py'}],
 'summary': 'Source: '
            'https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py '
            '(MistralAttention).',
 'adaptation_notes': ['static grouped-query attention with fewer KV heads, explicit KV expansion and fixed '
                      'causal sliding-window additive mask.']}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, dim: int, q_heads: int, kv_heads: int, seq: int, window: int):
        super().__init__()
        self.qh = q_heads
        self.kh = kv_heads
        self.d = dim // q_heads
        self.rep = q_heads // kv_heads
        self.q = nn.Linear(dim, q_heads * self.d)
        self.k = nn.Linear(dim, kv_heads * self.d)
        self.v = nn.Linear(dim, kv_heads * self.d)
        self.o = nn.Linear(q_heads * self.d, dim)
        pos = torch.arange(seq)
        dist = pos[:, None] - pos[None, :]
        allowed = (dist >= 0) & (dist < window)
        self.register_buffer('mask', torch.where(allowed, torch.zeros(seq, seq), torch.full((seq, seq), -10000.0)), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.q(x).reshape(b, n, self.qh, self.d).transpose(1, 2)
        k = self.k(x).reshape(b, n, self.kh, self.d).transpose(1, 2)
        v = self.v(x).reshape(b, n, self.kh, self.d).transpose(1, 2)
        k = k.unsqueeze(2).expand(b, self.kh, self.rep, n, self.d).reshape(b, self.qh, n, self.d)
        v = v.unsqueeze(2).expand(b, self.kh, self.rep, n, self.d).reshape(b, self.qh, n, self.d)
        a = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.d ** (-0.5) + self.mask, dim=-1)
        return self.o(torch.matmul(a, v).transpose(1, 2).reshape(b, n, -1))
batch_size = __EKB_PARAM_batch_size__
seq = __EKB_PARAM_seq__
dim = __EKB_PARAM_dim__
q_heads = __EKB_PARAM_q_heads__
kv_heads = __EKB_PARAM_kv_heads__
window = __EKB_PARAM_window__

def get_inputs():
    return [torch.randn(batch_size, seq, dim)]

def get_init_inputs():
    return [dim, q_heads, kv_heads, seq, window]
