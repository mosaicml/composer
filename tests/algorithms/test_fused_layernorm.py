# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import numpy as np
import pytest
import torch

from composer.algorithms.fused_layernorm import FusedLayerNorm, apply_fused_layernorm
from composer.core.event import Event
from composer.core.state import State
from composer.loggers import Logger
from composer.models import BERTModel
from tests.common import SimpleConvModel
from tests.fixtures.synthetic_hf_state import make_dataset_configs, synthetic_hf_state_maker


@pytest.fixture()
def synthetic_bert_state():
    synthetic_config = make_dataset_configs(model_family=['bert'])[0]
    return synthetic_hf_state_maker(synthetic_config)


def assert_is_fln_instance(model: BERTModel):
    pytest.importorskip("apex")
    from apex.normalization.fused_layer_norm import FusedLayerNorm as APEXFusedLayerNorm
    assert isinstance(model.module.bert.encoder.layer[0].output.LayerNorm, APEXFusedLayerNorm)


@pytest.mark.filterwarnings(
    r"ignore:Metric `SpearmanCorrcoef` will save all targets and predictions in the buffer:UserWarning:torchmetrics")
def test_fused_layernorm_functional(synthetic_bert_state: Tuple):
    state, model, dataloader = synthetic_bert_state
    print("Model:", model)
    apply_fused_layernorm(state.model, state.optimizers)
    assert_is_fln_instance(state.model)


@pytest.mark.filterwarnings(
    r"ignore:Metric `SpearmanCorrcoef` will save all targets and predictions in the buffer:UserWarning:torchmetrics")
@pytest.mark.parametrize(
    "device",
    [pytest.param("gpu", marks=pytest.mark.gpu)],
)
def test_fused_layernorm_algorithm(synthetic_bert_state: Tuple, empty_logger: Logger, device: str):
    state, _, _ = synthetic_bert_state
    fused_layernorm = FusedLayerNorm()
    if device == "gpu":
        state.model = state.model.cuda()  # move the model to gpu

    assert isinstance(state.model, BERTModel)
    fused_layernorm.apply(Event.INIT, state, empty_logger)

    assert_is_fln_instance(state.model)
