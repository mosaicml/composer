# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.nn import LayerNorm

from composer.algorithms.low_precision_layernorm import LowPrecisionLayerNorm, apply_low_precision_layernorm
from composer.algorithms.low_precision_layernorm.low_precision_layernorm import LPLayerNorm
from composer.core import Event, State
from composer.loggers import Logger
from composer.models.huggingface import HuggingFaceModel
from composer.utils import get_device
from tests.common import device
from tests.common.datasets import dummy_bert_lm_dataloader, dummy_text_classification_dataloader
from tests.common.models import SimpleTransformerClassifier, configure_tiny_bert_hf_model


def assert_is_lpln_instance(model):
    pytest.importorskip('transformers')

    # When checking modules of a HuggingFace model, we need to parse the model object it wraps
    # This is not necessary for SimpleTransformerClassifier models.
    if isinstance(model, HuggingFaceModel):
        model = model.model

    # ensure that within the entire model, no PyTorch LayerNorm exists, and at least one LPLN does.
    assert model.modules is not None, 'model has .modules method'
    for module_class in model.modules():
        if isinstance(module_class, LayerNorm):
            assert isinstance(module_class, LPLayerNorm)

    assert any(isinstance(module_class, LPLayerNorm) for module_class in model.modules())


@device('gpu')
@pytest.mark.parametrize('model,dataloader', [
    (configure_tiny_bert_hf_model, dummy_bert_lm_dataloader),
    (SimpleTransformerClassifier, dummy_text_classification_dataloader),
])
def test_low_precision_layernorm_functional(model, dataloader, device: str):
    model = model()

    # Remove biases and weights from some LayerNorms to test LPLN robustness
    if isinstance(model, SimpleTransformerClassifier):
        model.module[0].net[1].layers[0].norm1.bias = None  # type: ignore
        model.module[0].net[1].layers[0].norm2.weight = None  # type: ignore
        model.module[0].net[1].layers[0].norm2.bias = None  # type: ignore

    dataloader = dataloader()
    state = State(
        model=model,
        rank_zero_seed=0,
        run_name='run_name',
        device=get_device(device),
        dataloader=dataloader,
        dataloader_label='train',
        max_duration='1ep',
        precision='amp_fp16',
    )
    if device == 'gpu':
        state.model = state.model.cuda()  # move the model to gpu

    apply_low_precision_layernorm(state.model, state._precision, state.optimizers)
    assert_is_lpln_instance(state.model)


@device('gpu')
@pytest.mark.parametrize('model,dataloader', [
    (configure_tiny_bert_hf_model, dummy_bert_lm_dataloader),
    (SimpleTransformerClassifier, dummy_text_classification_dataloader),
])
def test_low_precision_layernorm_algorithm(model, dataloader, empty_logger: Logger, device: str):
    model = model()

    # Remove biases and weights from some LayerNorms to test LPLN robustness
    if isinstance(model, SimpleTransformerClassifier):
        model.module[0].net[1].layers[0].norm1.bias = None  # type: ignore
        model.module[0].net[1].layers[0].norm2.weight = None  # type: ignore
        model.module[0].net[1].layers[0].norm2.bias = None  # type: ignore

    dataloader = dataloader()
    state = State(
        model=model,
        rank_zero_seed=0,
        run_name='run_name',
        device=get_device(device),
        dataloader=dataloader,
        dataloader_label='train',
        max_duration='1ep',
        precision='amp_fp16',
    )
    low_precision_layernorm = LowPrecisionLayerNorm()
    if device == 'gpu':
        state.model = state.model.cuda()  # move the model to gpu

    low_precision_layernorm.apply(Event.INIT, state, empty_logger)

    assert_is_lpln_instance(state.model)
