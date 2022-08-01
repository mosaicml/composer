# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import pytest
import torch
from torch.nn.functional import gelu, relu

from composer.algorithms.gated_linear_units import GatedLinearUnits, apply_gated_linear_units
from composer.algorithms.gated_linear_units.gated_linear_unit_layers import BERTGatedFFOutput
from composer.core.event import Event
from composer.loggers import Logger
from tests.fixtures.synthetic_hf_state import make_dataset_configs, synthetic_hf_state_maker


def _layernorm(input_tensor, layernorm_eps):
    mean = torch.mean(input_tensor, dim=-1, keepdim=True)
    var = torch.square(input_tensor - mean).mean(dim=-1, keepdim=True)
    return (input_tensor - mean) / torch.sqrt(var + layernorm_eps)


@pytest.mark.parametrize('batch_size', [1])
@pytest.mark.parametrize('seq_length', [128, 512])
@pytest.mark.parametrize('d_embed', [768])
@pytest.mark.parametrize('d_ff', [3072])
@pytest.mark.parametrize('dropout_rate', [0.0])
@pytest.mark.parametrize('act_fn', [relu, gelu])
@pytest.mark.parametrize('layernorm_eps', [1e-6])
def test_glu_outputs(batch_size, seq_length, d_embed, d_ff, dropout_rate, act_fn, layernorm_eps):
    gated_ff = BERTGatedFFOutput(d_embed=d_embed,
                                 d_ff=d_ff,
                                 dropout_rate=dropout_rate,
                                 act_fn=act_fn,
                                 layernorm_eps=layernorm_eps,
                                 gated_layer_bias=False,
                                 non_gated_layer_bias=False)
    hidden_states = torch.rand(batch_size, seq_length, d_embed)
    residual_connection = torch.zeros_like(hidden_states)
    model_output = gated_ff(hidden_states, residual_connection)

    # get rid of the batch dimension when computing the result by hand
    hidden_states = hidden_states[1:]
    manual_output = torch.matmul(hidden_states, gated_ff.gated_layer.weight.transpose(0, 1))
    manual_output = act_fn(manual_output)
    manual_output = manual_output * torch.matmul(hidden_states, gated_ff.non_gated_layer.weight.transpose(0, 1))
    manual_output = torch.matmul(manual_output, gated_ff.wo.weight.transpose(0, 1)) + gated_ff.wo.bias
    manual_output = _layernorm(manual_output + residual_connection, layernorm_eps)
    assert torch.allclose(manual_output, model_output)


@pytest.fixture()
def synthetic_bert_state():
    synthetic_config = make_dataset_configs(model_family=['bert'])[0]
    return synthetic_hf_state_maker(synthetic_config)


def assert_is_glu_instance(model):
    pytest.importorskip('transformers')
    from transformers import BertForMaskedLM, BertForSequenceClassification
    from transformers.models.bert.modeling_bert import BertOutput

    assert isinstance(model, BertForMaskedLM) or isinstance(model, BertForSequenceClassification)
    # ensure that within the entire model, no BertOutput exists, and at least one BERTGatedFFOutput does.
    assert model.modules is not None, 'model has .modules method'
    for module_class in model.modules():
        assert not isinstance(
            module_class, BertOutput
        ), 'A transformers.models.bert.modeling_bert.BertOutput should not be found in the model after surgery is applied.'

    assert any(
        isinstance(module_class, BERTGatedFFOutput) for module_class in model.modules()
    ), 'composer.algorithms.gated_linear_units.gated_linear_unit_layers.BERTGatedFFOutput is not found in the post-surgery model.'


def test_gated_linear_units_functional(synthetic_bert_state: Tuple):
    state, _, _ = synthetic_bert_state
    apply_gated_linear_units(state.model, state.optimizers)
    assert_is_glu_instance(state.model.model)


def test_gated_linear_units_algorithm(synthetic_bert_state: Tuple, empty_logger: Logger):
    pytest.importorskip('transformers')
    from transformers import BertForMaskedLM, BertForSequenceClassification
    state, _, _ = synthetic_bert_state
    gated_linear_units = GatedLinearUnits()

    # state.model wrapped in HuggingFaceModel wrapped
    assert isinstance(state.model.model, BertForMaskedLM) or isinstance(state.model.model,
                                                                        BertForSequenceClassification)
    gated_linear_units.apply(Event.INIT, state, empty_logger)

    assert_is_glu_instance(state.model.model)
