# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import torch


class IdentityLayer(torch.nn.Module):
    """
    Defines a no-op module.
    """

    def forward(self, hidden_states):
        """
        A no-op forward that just returns the input.
        """
        return hidden_states


class BERTGatedFFOutput(torch.nn.Module):
    """
    Defines a single feed-forward block that uses `Gated Linear Units <https://arxiv.org/abs/2002.05202>`_.

    Args:
        d_embed (int): The input dimension for the feed-forward network.
        d_ff (int): The hidden dimension for the feed-forward network.
        dropout_rate (float): The dropout rate to use between the two projection matricies in the feed-forward block.
        act_fn (Callable): The activation function to use in the feed-forward network.
        layernorm_eps (float): The epsilon term to use in the LayerNorm operator. Useful for when the variance is small.
        gated_layer_bias (bool): Whether to use a bias term in the gated projection matrix.
        non_gated_layer_bias (bool): Whether to use a bias term in teh non-gated projection matrix.
    """

    def __init__(self,
                 d_embed: int,
                 d_ff: int,
                 dropout_rate: float,
                 act_fn: Callable,
                 layernorm_eps: float,
                 gated_layer_bias: bool = False,
                 non_gated_layer_bias: bool = False):
        super().__init__()
        self.gated_layer = torch.nn.Linear(d_embed, d_ff, bias=gated_layer_bias)
        self.non_gated_layer = torch.nn.Linear(d_embed, d_ff, bias=non_gated_layer_bias)
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
        hidden_states = self.act(self.gated_layer(hidden_states)) * self.non_gated_layer(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states
