# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest
from torch.utils.data import Dataset

from composer.algorithms import (AlgorithmHparams, AlibiHparams, CutMixHparams, StochasticDepthHparams,
                                 algorithm_registry)
from composer.core.algorithm import Algorithm
from composer.models.base import ComposerModel
from tests.algorithms.algorithm_settings import get_settings

default_required_fields = {
    AlibiHparams: {
        'position_embedding_attribute': 'module.transformer.wpe',
        'attention_module_name': 'transformers.models.gpt2.modeling_gpt2.GPT2Attention',
        'attr_to_replace': '_attn',
        'alibi_attention': 'composer.algorithms.alibi._gpt2_alibi._attn',
        'mask_replacement_function': 'composer.algorithms.alibi._gpt2_alibi.enlarge_mask',
    },
    CutMixHparams: {
        'num_classes': 1000
    },
    StochasticDepthHparams: {
        'target_layer_name': 'ResNetBottleneck',
    },
}


@pytest.fixture
def registry():
    return algorithm_registry.get_algorithm_registry()


@pytest.mark.parametrize("name", algorithm_registry.list_algorithms())
def test_algorithm_registry(name, registry):
    # create the hparams object
    hparams_class = registry[name]

    kwargs = default_required_fields.get(hparams_class, dict())
    hparams = hparams_class(**kwargs)
    assert isinstance(hparams, AlgorithmHparams)
    assert dataclasses.is_dataclass(hparams)

    algorithm = algorithm_registry.get_algorithm(hparams)
    assert isinstance(algorithm, Algorithm)


@pytest.mark.parametrize("name", algorithm_registry.list_algorithms())
def test_algorithm_settings(name):
    if name in ('alibi', 'seq_length_warmup', 'factorize', 'no_op_model', 'scale_schedule'):
        pytest.skip()

    setting = get_settings(name)
    if setting is None:
        pytest.skip()

    assert isinstance(setting['algorithm'], Algorithm)
    assert isinstance(setting['model'], ComposerModel)
    assert isinstance(setting['dataset'], Dataset)
