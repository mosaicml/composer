# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch

from composer.algorithms import Factorize
from composer.algorithms.factorize import FactorizedConv2d, FactorizedLinear
from composer.algorithms.factorize.factorize import LOG_NUM_CONV2D_REPLACEMENTS_KEY, LOG_NUM_LINEAR_REPLACEMENTS_KEY
from composer.core import Event, State
from composer.loggers import Logger
from composer.models import HuggingFaceModel
from composer.utils import module_surgery
from tests.common import SimpleConvModel, SimpleTransformerClassifier
from tests.common.datasets import dummy_tiny_bert_lm_batch, dummy_transformer_classifier_batch
from tests.common.models import configure_tiny_bert_hf_model


def create_state(minimal_state: State, model):
    minimal_state.model = model
    return minimal_state


def create_algo_instance(replace_convs, replace_linears):
    return Factorize(factorize_convs=replace_convs,
                     factorize_linears=replace_linears,
                     min_channels=1,
                     latent_channels=2,
                     min_features=1,
                     latent_features=2)


@pytest.mark.parametrize('model_cls, model_params', [(SimpleConvModel, (3, 100)), (SimpleTransformerClassifier, ()),
                                                     (configure_tiny_bert_hf_model, ())])
@pytest.mark.parametrize('replace_convs', [False, True])
@pytest.mark.parametrize('replace_linears', [False, True])
def test_factorize_surgery(minimal_state: State, model_cls, model_params, empty_logger: Logger, replace_convs: bool,
                           replace_linears: bool):
    model = model_cls(*model_params)
    state = create_state(minimal_state, model)

    if (isinstance(model, SimpleTransformerClassifier) or isinstance(model, HuggingFaceModel)) and replace_convs:
        pytest.skip('Skipping: NLP models do not contain conv layers.')

    algo_instance = create_algo_instance(replace_convs, replace_linears)

    num_conv_layers = module_surgery.count_module_instances(state.model, torch.nn.Conv2d)
    num_linear_layers = module_surgery.count_module_instances(state.model, torch.nn.Linear)

    algo_instance.apply(event=Event.INIT, state=state, logger=empty_logger)

    # Each Conv2d/Linear is either unmodified or replaced with a factorized version,
    # which contains two Conv2d/Linears submodules.
    if algo_instance.factorize_convs:
        num_factorized_layers = module_surgery.count_module_instances(state.model, FactorizedConv2d)
        num_non_factorized_layers = module_surgery.count_module_instances(state.model,
                                                                          torch.nn.Conv2d) - 2 * num_factorized_layers
        assert num_conv_layers == num_factorized_layers + num_non_factorized_layers
        assert num_factorized_layers > 0

    if algo_instance.factorize_linears:
        num_factorized_layers = module_surgery.count_module_instances(state.model, FactorizedLinear)
        num_non_factorized_layers = module_surgery.count_module_instances(state.model,
                                                                          torch.nn.Linear) - 2 * num_factorized_layers
        assert num_linear_layers == num_factorized_layers + num_non_factorized_layers
        assert num_factorized_layers > 0


@pytest.mark.parametrize('model_cls, model_params, batch',
                         [(SimpleConvModel, (3, 100), (torch.Tensor(64, 3, 32, 32), torch.Tensor())),
                          (SimpleTransformerClassifier, (), dummy_transformer_classifier_batch()),
                          (configure_tiny_bert_hf_model, (), dummy_tiny_bert_lm_batch())])
@pytest.mark.parametrize('replace_convs', [False, True])
@pytest.mark.parametrize('replace_linears', [False, True])
def test_forward_shape(minimal_state: State, model_cls, model_params, empty_logger: Logger, batch, replace_convs,
                       replace_linears):
    model = model_cls(*model_params)

    if (isinstance(model, SimpleTransformerClassifier) or isinstance(model, HuggingFaceModel)) and replace_convs:
        pytest.skip('Skipping: NLP models do not contain conv layers.')

    if isinstance(model, SimpleTransformerClassifier):
        pytest.xfail(
            'Factorize does not support torch.nn.MultiheadAttention layers, which are part of the SimpleTransformerClassifier.'
        )

    state = create_state(minimal_state, model)
    algo_instance = create_algo_instance(replace_convs, replace_linears)
    output = state.model.forward(batch)

    algo_instance.apply(event=Event.INIT, state=state, logger=empty_logger)
    new_output = state.model.forward(batch)

    if isinstance(model, HuggingFaceModel):
        assert output.logits.size() == new_output.logits.size()
    else:
        assert output.size() == new_output.size()


@pytest.mark.parametrize('model_cls, model_params', [(SimpleConvModel, (3, 100)), (SimpleTransformerClassifier, ()),
                                                     (configure_tiny_bert_hf_model, ())])
@pytest.mark.parametrize('replace_convs', [False, True])
@pytest.mark.parametrize('replace_linears', [False, True])
def test_algorithm_logging(minimal_state: State, model_cls, model_params, replace_convs, replace_linears):
    model = model_cls(*model_params)
    state = create_state(minimal_state, model)

    if (isinstance(model, SimpleTransformerClassifier) or isinstance(model, HuggingFaceModel)) and replace_convs:
        pytest.skip('Skipping: NLP models do not contain conv layers.')

    algo_instance = create_algo_instance(replace_convs, replace_linears)

    logger_mock = Mock()

    algo_instance.apply(Event.INIT, state, logger=logger_mock)

    factorize_convs = algo_instance.factorize_convs
    factorize_linears = algo_instance.factorize_linears

    mock_obj = logger_mock.log_hyperparameters

    if factorize_convs:
        num_factorized_convs = module_surgery.count_module_instances(state.model, FactorizedConv2d)
        mock_obj.assert_any_call({LOG_NUM_CONV2D_REPLACEMENTS_KEY: num_factorized_convs})
    if factorize_linears:
        num_factorized_linears = module_surgery.count_module_instances(state.model, FactorizedLinear)
        mock_obj.assert_any_call({LOG_NUM_LINEAR_REPLACEMENTS_KEY: num_factorized_linears})

    target_count = 0
    target_count += 1 if factorize_convs else 0
    target_count += 1 if factorize_linears else 0
    assert mock_obj.call_count == target_count
