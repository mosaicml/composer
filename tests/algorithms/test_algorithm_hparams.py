from typing import Type

import pytest

import composer.algorithms
from composer.algorithms import AlgorithmHparams
from composer.algorithms.algorithm_registry import registry as algorithm_registry
from composer.core import Algorithm
from tests.algorithms.algorithm_settings import get_alg_kwargs
from tests.common import assert_is_constructable_from_yaml, assert_registry_contains_entry, get_all_subclasses_in_module


@pytest.mark.parametrize("alg_hparams_cls", get_all_subclasses_in_module(composer.algorithms, AlgorithmHparams))
def test_all_algs_in_registry(alg_hparams_cls: Type[AlgorithmHparams]):
    assert_registry_contains_entry(alg_hparams_cls, algorithm_registry)


@pytest.mark.xfail(reason="This test depends on AutoYAHP")
@pytest.mark.parametrize("alg_cls", get_all_subclasses_in_module(composer.algorithms, Algorithm))
def test_algs_are_constructable(alg_cls: Type[Algorithm]):
    kwargs = get_alg_kwargs(alg_cls)
    if kwargs is None:
        pytest.xfail(f"Missing settings for algorithm {alg_cls.__name__}")
    assert_is_constructable_from_yaml(alg_cls, kwargs, expected=alg_cls)
