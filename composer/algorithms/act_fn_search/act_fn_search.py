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

from .norms import RMSNorm

log = logging.getLogger(__name__)


@dataclass
class ActFnSearchHparams(AlgorithmHparams):
    """See :class:`Primer`"""
    act_fn_name: str = hp.required("The name of the activation function to use.")
    use_gated: bool = hp.required("Whether to use a GLU unit or a regular unit.")
    use_rmsnorm: bool = hp.required("Whether to use RMSNorm instead of LayerNorm.")

    def initialize_object(self) -> "Primer":
        return ActFnSearch(**asdict(self))


def apply_act_fn(model: torch.nn.Module, act_fn_name: str, use_gated: bool, use_rmsnorm: bool) -> None:
    act_fns = {
        "squared_relu": lambda x: relu(x)**2,
        "fast_gelu": transformers.activations.gelu_fast,
        "gelu": torch.nn.functional.gelu,
        "relu": lambda x: relu(x),
        "swish": transformers.activations.silu,
        "no_replacement": None,
    }
    act_fn = act_fns[act_fn_name]

    d_embeds = []
    layernorm_eps = []
    for idx in range(len(model.module.bert.encoder.layer)):
        bert_layer = model.module.bert.encoder.layer[idx]
        d_embeds.append(bert_layer.intermediate.dense.in_features)
        layernorm_eps.append(bert_layer.output.LayerNorm.eps)
    assert len(set(d_embeds)) == 1
    assert len(set(layernorm_eps)) == 1
    d_embed = d_embeds[0]
    layernorm_eps = layernorm_eps[0]

    if act_fn is not None:
        if not use_gated:
            for idx in range(len(model.module.bert.encoder.layer)):
                model.module.bert.encoder.layer[idx].intermediate.intermediate_act_fn = act_fn
        else:
            for idx in range(len(model.module.bert.encoder.layer)):
                # TODO: implement ReLU, GeLU, GEGeLU, ReGLU, and SwiGLU
                d_embed = model.module.bert.encoder.layer[idx].intermediate.dense.in_features
                d_ff = model.module.bert.encoder.layer[idx].intermediate.dense.out_features
                # scale down d_ff by 1/3 in order to maintain equal number of parameters
                d_ff = round((2.0 / 3.0) * d_ff)
                dropout_rate = model.module.bert.encoder.layer[idx].output.dropout.p
                layernorm_eps = model.module.bert.encoder.layer[idx].output.LayerNorm.eps
                model.module.bert.encoder.layer[idx].intermediate = DummyBERTIntermediateOutput()
                model.module.bert.encoder.layer[idx].output = BERTGatedOutput(d_embed=d_embed,
                                                                              d_ff=d_ff,
                                                                              dropout_rate=dropout_rate,
                                                                              act_fn=act_fn,
                                                                              layernorm_eps=layernorm_eps)

    if use_rmsnorm:
        policy = {torch.nn.LayerNorm: lambda x, module_index: RMSNorm(dim=d_embed, eps=layernorm_eps)}
        surgery.replace_module_classes(model=model, policies=policy)
    print(model)


class DummyBERTIntermediateOutput(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        return hidden_states


class BERTGatedOutput(torch.nn.Module):

    def __init__(self, d_embed, d_ff, dropout_rate, act_fn, layernorm_eps):
        super().__init__()
        self.wi_0 = torch.nn.Linear(d_embed, d_ff)
        self.wi_1 = torch.nn.Linear(d_embed, d_ff)
        self.wo = torch.nn.Linear(d_ff, d_embed)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.act = act_fn
        self.LayerNorm = torch.nn.LayerNorm(d_embed, eps=layernorm_eps)

    def forward(self, hidden_states, input_tensor):
        # compute the activation
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # compute the gate
        hidden_linear = self.wi_1(hidden_states)
        hidden_combo = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_combo)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ActFnSearch(Algorithm):

    def __init__(self, act_fn_name: str, use_gated: bool, use_rmsnorm: bool) -> None:
        self.act_fn_name = act_fn_name
        self.use_gated = use_gated
        self.use_rmsnorm = use_rmsnorm

    def match(self, event: Event, state: State) -> bool:
        """ Runs on Event.INIT
        """
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """ Replace model's existing attention mechanism with AliBi
        """

        if event == Event.INIT:
            assert state.model is not None
            apply_act_fn(state.model,
                         act_fn_name=self.act_fn_name,
                         use_gated=self.use_gated,
                         use_rmsnorm=self.use_rmsnorm)
