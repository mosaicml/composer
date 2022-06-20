# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import torch


class DummyBERTIntermediateOutput(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        return hidden_states


class BERTGatedOutput(torch.nn.Module):

    def __init__(self, d_embed, d_ff, dropout_rate, act_fn, layernorm_eps, wi_0_bias, wi_1_bias):
        super().__init__()
        self.wi_0 = torch.nn.Linear(d_embed, d_ff, bias=wi_0_bias)
        self.wi_1 = torch.nn.Linear(d_embed, d_ff, bias=wi_1_bias)
        self.wo = torch.nn.Linear(d_ff, d_embed)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.act = act_fn
        self.layernorm = torch.nn.LayerNorm(d_embed, eps=layernorm_eps)

    def forward(self, hidden_states, input_tensor):
        # compute the activation
        hidden_states = self.act(self.wi_0(hidden_states)) * self.wi_1(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states
