# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import pytest

from composer.algorithms.gated_linear_units import GatedLinearUnits, apply_gated_linear_units
from composer.algorithms.gated_linear_units.gated_linear_unit_layers import BERTGatedFFOutput
from composer.core.event import Event
from composer.loggers import Logger
from composer.models import BERTModel
from tests.fixtures.synthetic_hf_state import make_dataset_configs, synthetic_hf_state_maker


@pytest.fixture()
def synthetic_bert_state():
    synthetic_config = make_dataset_configs(model_family=['bert'])[0]
    return synthetic_hf_state_maker(synthetic_config)


def assert_is_glu_instance(model: BERTModel):
    pytest.importorskip('transformers')
    from transformers.models.bert.modeling_bert import BertOutput

    # ensure that within the entire model, no BertOutput exists, and at least one BERTGatedFFOutput does.
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
    assert_is_glu_instance(state.model)


def test_gated_linear_units_algorithm(synthetic_bert_state: Tuple, empty_logger: Logger):
    state, _, _ = synthetic_bert_state
    gated_linear_units = GatedLinearUnits()

    assert isinstance(state.model, BERTModel)
    gated_linear_units.apply(Event.INIT, state, empty_logger)

    assert_is_glu_instance(state.model)
