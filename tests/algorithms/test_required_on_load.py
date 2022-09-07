# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Type

import pytest

from composer import algorithms
from composer.core import Algorithm, Time, TimeUnit  # type: ignore imports used in `eval(representation)`


def initialize_algorithm(algo_cls: Type):
    """Initialize algorithm with dummy values."""
    if algo_cls == algorithms.Alibi:
        return algo_cls(max_sequence_length=1)
    elif algo_cls == algorithms.StochasticDepth:
        return algo_cls(target_layer_name='ResNetBottleneck')
    elif algo_cls == algorithms.FusedLayerNorm:
        pytest.importorskip('apex')
        return algo_cls()
    elif algo_cls == algorithms.GatedLinearUnits:
        pytest.importorskip('transformers')
        return algo_cls()
    else:
        return algo_cls()


@pytest.mark.parametrize('algo_name', algorithms.__all__)
def test_required_on_load_has_repr(algo_name: str):
    algo_cls = getattr(algorithms, algo_name)
    if issubclass(algo_cls, Algorithm) and algo_cls.required_on_load():
        representation = repr(initialize_algorithm(algo_cls))
        # Default repr prints memory address
        assert 'at 0x' not in representation
        eval(f'algorithms.{representation}')
