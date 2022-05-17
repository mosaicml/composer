from typing import Type

import pytest

import composer.algorithms
from composer.algorithms.algorithm_hparams import algorithm_registry
from composer.core import Algorithm
from tests.common import assert_is_constructable_from_yaml, assert_registry_contains_entry, get_all_subclasses_in_module


@pytest.mark.parametrize("alg_cls", get_all_subclasses_in_module(composer.algorithms, Algorithm))
def test_all_algs_in_registry(alg_cls: Type[Algorithm]):
    assert_registry_contains_entry(alg_cls, algorithm_registry)


@pytest.mark.parametrize("alg_cls", get_all_subclasses_in_module(composer.algorithms, Algorithm))
def test_algs_are_constructable(alg_cls: Type[Algorithm]):
    assert_is_constructable_from_yaml(alg_cls, {}, expected=alg_cls)
