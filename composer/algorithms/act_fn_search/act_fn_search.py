# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from types import MethodType, ModuleType
from typing import Any, Callable, List, Optional, Union

import torch
import transformers
import yahp as hp
from torch.nn.functional import relu

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State, surgery

log = logging.getLogger(__name__)


@dataclass
class ActFnHparams(AlgorithmHparams):
    """See :class:`Primer`"""
    act_fn_name: str = hp.required("The name of the activation function to use.")

    def initialize_object(self) -> "Primer":
        return Primer(**asdict(self))


def make_layerwise_convs(model, idx, dim_per_head, n_heads, kernel_size):
    model.module.transformer.h[idx].q_dconv = CausalDepthwiseConv(dim_per_head, n_heads, kernel_size=kernel_size)
    model.module.transformer.h[idx].k_dconv = CausalDepthwiseConv(dim_per_head, n_heads, kernel_size=kernel_size)
    model.module.transformer.h[idx].v_dconv = CausalDepthwiseConv(dim_per_head, n_heads, kernel_size=kernel_size)

    orig_attn_fn = model.module.transformer.h[idx].attn._attn

    def dconv_attn(query, key, value, attention_mask=None, head_mask=None):
        # query shape is (bs x nhead x seq_len x head_dim)
        # the dconv expects (bs x seq_len x nhead x head_dim)
        query = model.module.transformer.h[idx].q_dconv(query.transpose(1, 2)).transpose(1, 2)
        key = model.module.transformer.h[idx].k_dconv(key.transpose(1, 2)).transpose(1, 2)
        value = model.module.transformer.h[idx].v_dconv(value.transpose(1, 2)).transpose(1, 2)
        attn = orig_attn_fn(query, key, value, attention_mask=attention_mask, head_mask=head_mask)
        return attn

    model.module.transformer.h[idx].attn._attn = dconv_attn


def apply_primer(model: torch.nn.Module, act_fn_name: str) -> None:
    act_fns = {"squared_relu": lambda x: relu(x)**2, "fast_gelu": transformers.activations.gelu_fast, "relu": lambda x: relu(x), "swish": transformers.activations.silu}
    act_fn = act_fns[act_fn_name]

    if "gated" not in act_fn_name:
        for idx in range(len(model.module.bert.encoder.layer)):
            model.module.bert.encoder.layer[idx].intermediate.intermediate_act_fn = act_fn
    else:
        # TODO: implement ReLU, GeLU, GEGeLU, ReGLU, and SwiGLU


# adapted from GPT Neo-X
class CausalDepthwiseConv(torch.nn.Module):

    def __init__(self, dim_per_head, n_heads, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = torch.nn.Parameter(torch.empty(size=(kernel_size, n_heads, dim_per_head)))
        # weird init from https://github.com/google-research/google-research/blob/3e1a06764ff52e33e3523d82ae836441df701c5d/primer/t5_models.py#L35
        torch.nn.init.constant_(self.weight, 0.5 / kernel_size)
        torch.nn.init.constant_(self.weight[0], 0.5)

    def forward(self, x, seq_dim=1):
        # x should be [b, s, np, hp]
        ret = x * self.weight[0]
        if self.kernel_size == 3:
            shift_distance = 1
            x = shift(x, 1, dim=seq_dim)
            ret += x * self.weight[shift_distance]

            shift_distance = 2
            x = shift(x, 1, dim=seq_dim)
            ret += x * self.weight[shift_distance]
        else:
            for shift_distance in range(1, self.kernel_size):
                x = shift(x, 1, dim=seq_dim)
                ret += x * self.weight[shift_distance]
        return ret

class DummyBERTIntermediateOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        return hidden_states

class BERTGatedOutput(torch.nn.Module):
    def __init__(self, d_embed, d_ff, dropout_rate, act_fn, layernorm_eps):
        self.wi_0 = torch.nn.Linear(d_embed, d_ff)
        self.wi_1 = torch.nn.Linear(d_embed, d_ff)
        self.wo = torch.nn.Linear(d_ff, d_embed)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.act = act_fn
        self.LayerNorm = torch.nn.LayerNorm(d_embed, eps=layernorm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_linear)
        hidden_combo = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)


class Primer(Algorithm):

    def __init__(self, use_squared_relu: bool, use_dconv: bool, use_every_n_layers: int) -> None:
        self.use_squared_relu = use_squared_relu
        self.use_dconv = use_dconv
        self.use_every_n_layers = use_every_n_layers

    def match(self, event: Event, state: State) -> bool:
        """ Runs on Event.INIT
        """
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """ Replace model's existing attention mechanism with AliBi
        """

        if event == Event.INIT:
            assert state.model is not None
            apply_primer(state.model,
                         use_squared_relu=self.use_squared_relu,
                         use_dconv=self.use_dconv,
                         use_every_n_layers=self.use_every_n_layers)
