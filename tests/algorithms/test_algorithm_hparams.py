# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Type

import pytest

import composer.algorithms
from composer.algorithms.algorithm_hparams_registry import algorithm_registry
from composer.core import Algorithm
from tests.algorithms.algorithm_settings import get_alg_kwargs, get_algs_with_marks
from tests.common import get_module_subclasses
from tests.common.hparams import assert_in_registry, construct_from_yaml


@pytest.mark.parametrize('alg_cls', get_algs_with_marks())
def test_algs_are_constructable(alg_cls: Type[Algorithm]):
    assert isinstance(alg_cls(**get_alg_kwargs(alg_cls)), Algorithm)


@pytest.mark.parametrize('alg_cls', get_module_subclasses(composer.algorithms, Algorithm))
def test_all_algs_in_registry(alg_cls: Type[Algorithm]):
    assert_in_registry(alg_cls, algorithm_registry)


@pytest.mark.parametrize('alg_cls', get_algs_with_marks())
def test_algs_load_from_yaml(alg_cls: Type[Algorithm]):
    kwargs = get_alg_kwargs(alg_cls)
    if kwargs is None:
        pytest.xfail(f'Missing settings for algorithm {alg_cls.__name__}')
    assert isinstance(construct_from_yaml(alg_cls, kwargs), alg_cls)
