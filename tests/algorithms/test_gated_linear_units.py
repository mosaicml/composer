# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn.functional import gelu, relu

from composer.algorithms.gated_linear_units import GatedLinearUnits, apply_gated_linear_units
from composer.algorithms.gated_linear_units.gated_linear_unit_layers import BERTGatedFFOutput
from composer.core import Event, State
from composer.devices import DeviceCPU
from composer.loggers import Logger
from tests.common.datasets import dummy_bert_lm_dataloader, dummy_text_classification_dataloader
from tests.common.models import SimpleTransformerClassifier, configure_tiny_bert_hf_model


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
    gated_ff = BERTGatedFFOutput(
        d_embed=d_embed,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
        act_fn=act_fn,
        layernorm_eps=layernorm_eps,
        gated_layer_bias=False,
        non_gated_layer_bias=False,
    )
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


def assert_is_glu_instance(model):
    pytest.importorskip('transformers')
    from transformers import BertForMaskedLM, BertForSequenceClassification
    from transformers.models.bert.modeling_bert import BertOutput

    assert isinstance(model, BertForMaskedLM) or isinstance(model, BertForSequenceClassification)
    # ensure that within the entire model, no BertOutput exists, and at least one BERTGatedFFOutput does.
    assert model.modules is not None, 'model has .modules method'
    for module_class in model.modules():
        assert not isinstance(
            module_class,
            BertOutput,
        ), 'A transformers.models.bert.modeling_bert.BertOutput should not be found in the model after surgery is applied.'

    assert any(
        isinstance(module_class, BERTGatedFFOutput) for module_class in model.modules()
    ), 'composer.algorithms.gated_linear_units.gated_linear_unit_layers.BERTGatedFFOutput is not found in the post-surgery model.'


@pytest.mark.parametrize(
    'model,dataloader',
    [
        (configure_tiny_bert_hf_model, dummy_bert_lm_dataloader),
        (
            pytest.param(
                SimpleTransformerClassifier,
                dummy_text_classification_dataloader,
                marks=pytest.mark.xfail(reason='Gated Linear Units does not currently support non-HuggingFace models'),
            )
        ),
    ],
)
def test_gated_linear_units_functional(model, dataloader):
    model = model()
    dataloader = dataloader()
    state = State(
        model=model,
        rank_zero_seed=0,
        run_name='run_name',
        device=DeviceCPU(),
        dataloader=dataloader,
        dataloader_label='train',
        max_duration='1ep',
    )
    apply_gated_linear_units(state.model, state.optimizers)
    assert_is_glu_instance(state.model.model)


@pytest.mark.parametrize(
    'model,dataloader',
    [
        (configure_tiny_bert_hf_model, dummy_bert_lm_dataloader),
        (
            pytest.param(
                SimpleTransformerClassifier,
                dummy_text_classification_dataloader,
                marks=pytest.mark.xfail(reason='Gated Linear Units does not currently support non-HuggingFace models'),
            )
        ),
    ],
)
def test_gated_linear_units_algorithm(model, dataloader, empty_logger: Logger):
    pytest.importorskip('transformers')
    from transformers import BertForMaskedLM, BertForSequenceClassification

    model = model()
    dataloader = dataloader()
    state = State(
        model=model,
        rank_zero_seed=0,
        run_name='run_name',
        device=DeviceCPU(),
        dataloader=dataloader,
        dataloader_label='train',
        max_duration='1ep',
    )

    gated_linear_units = GatedLinearUnits()

    # state.model wrapped in HuggingFaceModel wrapped
    assert isinstance(
        state.model.model,
        BertForMaskedLM,
    ) or isinstance(state.model.model, BertForSequenceClassification)
    gated_linear_units.apply(Event.INIT, state, empty_logger)

    assert_is_glu_instance(state.model.model)
