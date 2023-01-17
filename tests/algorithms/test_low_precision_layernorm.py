# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import pytest
from torch.nn import LayerNorm

from composer.algorithms.low_precision_layernorm import LowPrecisionLayerNorm, apply_low_precision_layernorm
from composer.algorithms.low_precision_layernorm.low_precision_layernorm import LPLayerNorm
from composer.core.event import Event
from composer.core.precision import Precision
from composer.loggers import Logger
from composer.models.huggingface import HuggingFaceModel
from tests.common import device
from tests.fixtures.synthetic_hf_state import (make_dataset_configs, synthetic_hf_state_maker,
                                               synthetic_simple_transformer_state_maker)


def make_synthetic_state(family, session):
    """Supported model families are 'simple_transformer', 'bert', 'gpt2', and 'bert_classification'.
    """
    if family == 'simple_transformer':
        return synthetic_simple_transformer_state_maker(session)
    synthetic_config = make_dataset_configs(model_family=[family])[0]
    return synthetic_hf_state_maker(synthetic_config, session)


def assert_is_lpln_instance(model):
    pytest.importorskip('transformers')

    # When checking modules of a HuggingFace model, we need to parse the model object it wraps
    # This is not necessary for SimpleTransformerClassifier models.
    if isinstance(model, HuggingFaceModel):
        model = model.model

    # ensure that within the entire model, no PyTorch LayerNorm exists, and at least one APEX FLN does.
    assert model.modules is not None, 'model has .modules method'
    for module_class in model.modules():
        if isinstance(module_class, LayerNorm):
            assert isinstance(
                module_class,
                LPLayerNorm), 'A standard torch.nn.LayerNorm should not be found in the model after surgery is applied.'

    assert any(
        isinstance(module_class, LPLayerNorm) for module_class in model.modules()
    ), 'composer.algorithms.low_precision_layernorm.low_precision_layernorm.LPLayerNorm is not found in the post-surgery model.'


@device('gpu')
@pytest.mark.parametrize('synthetic_state_family', [
    'bert',
    'simple_transformer',
])
def test_low_precision_layernorm_functional(synthetic_state_family: Tuple, device: str, request: pytest.FixtureRequest):
    state, _, _ = make_synthetic_state(synthetic_state_family, request.session)
    if device == 'gpu':
        state.model = state.model.cuda()  # move the model to gpu
    state._precision = Precision('amp_fp16')

    apply_low_precision_layernorm(state.model, state.optimizers, state._precision)
    assert_is_lpln_instance(state.model)


@device('gpu')
@pytest.mark.parametrize('synthetic_state_family', [
    'bert',
    'simple_transformer',
])
def test_low_precision_layernorm_algorithm(synthetic_state_family: Tuple, empty_logger: Logger, device: str,
                                           request: pytest.FixtureRequest):
    state, _, _ = make_synthetic_state(synthetic_state_family, request.session)
    state._precision = Precision('amp_fp16')
    low_precision_layernorm = LowPrecisionLayerNorm()
    if device == 'gpu':
        state.model = state.model.cuda()  # move the model to gpu

    low_precision_layernorm.apply(Event.INIT, state, empty_logger)

    assert_is_lpln_instance(state.model)
