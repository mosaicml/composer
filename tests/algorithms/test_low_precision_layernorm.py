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
from tests.common import device
from tests.fixtures.synthetic_hf_state import make_dataset_configs, synthetic_hf_state_maker


@pytest.fixture()
def synthetic_bert_state():
    synthetic_config = make_dataset_configs(model_family=['bert'])[0]
    return synthetic_hf_state_maker(synthetic_config)


def assert_is_lpln_instance(model):
    pytest.importorskip('transformers')
    from transformers import BertForMaskedLM, BertForSequenceClassification

    assert isinstance(model, BertForMaskedLM) or isinstance(model, BertForSequenceClassification)
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
def test_low_precision_layernorm_functional(synthetic_bert_state: Tuple, device: str):
    state, _, _ = synthetic_bert_state
    state._precision = Precision('amp')
    apply_low_precision_layernorm(state.model, state.optimizers, state._precision)
    assert_is_lpln_instance(state.model.model)


@device('gpu')
def test_low_precision_layernorm_algorithm(synthetic_bert_state: Tuple, empty_logger: Logger, device: str):
    pytest.importorskip('transformers')
    from transformers import BertForMaskedLM, BertForSequenceClassification

    state, _, _ = synthetic_bert_state
    state._precision = Precision('amp')
    low_precision_layernorm = LowPrecisionLayerNorm()
    if device == 'gpu':
        state.model = state.model.cuda()  # move the model to gpu

    # state.model wrapped in HuggingFaceModel wrapped
    assert isinstance(state.model.model, BertForMaskedLM) or isinstance(state.model.model,
                                                                        BertForSequenceClassification)
    low_precision_layernorm.apply(Event.INIT, state, empty_logger)

    assert_is_lpln_instance(state.model.model)
