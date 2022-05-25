# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Type

import pytest

import composer.algorithms
from composer.algorithms import AlgorithmHparams
from composer.algorithms.algorithm_registry import registry as algorithm_registry
from composer.core import Algorithm
from tests.algorithms.algorithm_settings import get_alg_kwargs
from tests.common import get_module_subclasses
from tests.common.hparams import assert_in_registry, assert_yaml_loads


@pytest.mark.parametrize("alg_hparams_cls", get_module_subclasses(composer.algorithms, AlgorithmHparams))
def test_all_algs_in_registry(alg_hparams_cls: Type[AlgorithmHparams]):
    assert_in_registry(alg_hparams_cls, algorithm_registry)


@pytest.mark.xfail(reason="This test depends on AutoYAHP")
@pytest.mark.parametrize("alg_cls", get_module_subclasses(composer.algorithms, Algorithm))
def test_algs_load_from_yaml(alg_cls: Type[Algorithm]):
    kwargs = get_alg_kwargs(alg_cls)
    if kwargs is None:
        pytest.xfail(f"Missing settings for algorithm {alg_cls.__name__}")
    assert_yaml_loads(alg_cls, kwargs, expected=alg_cls)
