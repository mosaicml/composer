# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import torch


class DummyBERTIntermediateOutput(torch.nn.Module):
    """
    Defines a no-op module to replace `transformers.models.bert.modeling_bert.BertIntermediate`.
    We absorb these layers into :class:`BERTGatedOutput`.
    """

    def forward(self, hidden_states):
        """
        A no-op forward that just returns the input.
        """
        return hidden_states


class BERTGatedOutput(torch.nn.Module):
    """
    Defines a single unit for a Gated Linear Unit, which substitutes a standard feed-forward layer in BERT.

    Args:
        d_embed (int): The input dimension for the feed-forward network.
        d_ff (int): The hidden dimension for the feed-forward network.
        dropout_rate (float): The dropout rate to use between the two projection matricies in the feed-forward block.
        act_fn (Callable): The activation function to use in the feed-forward network.
        layernorm_eps (float): The epsilon term to use in the LayerNorm operator. Useful for when the variance is small.
        wi_0_bias (bool): Whether to use a bias term in the gated projection matrix.
        wi_1_bias (bool): Whether to use a bias term in teh non-gated projection matrix.
    """

    def __init__(self,
                 d_embed: int,
                 d_ff: int,
                 dropout_rate: float,
                 act_fn: Callable,
                 layernorm_eps: float,
                 wi_0_bias: bool = False,
                 wi_1_bias: bool = False):
        super().__init__()
        self.wi_0 = torch.nn.Linear(d_embed, d_ff, bias=wi_0_bias)
        self.wi_1 = torch.nn.Linear(d_embed, d_ff, bias=wi_1_bias)
        self.wo = torch.nn.Linear(d_ff, d_embed)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.act = act_fn
        self.layernorm = torch.nn.LayerNorm(d_embed, eps=layernorm_eps)

    def forward(self, hidden_states, input_tensor):
        """
        Args:
            hidden_states (:class:`torch.Tensor`): The hidden states from the attention matrix.
            input_tensor (:class:`torch.Tensor`): The residual connection to add before the LayerNorm operator.
        """
        # compute the activation
        hidden_states = self.act(self.wi_0(hidden_states)) * self.wi_1(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states
