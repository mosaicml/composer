# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict

import pytest
from torch.utils.data import Dataset

from composer.core.algorithm import Algorithm
from composer.models.base import ComposerModel
from tests.algorithms.algorithm_settings import get_algorithm_parametrization


@pytest.mark.parametrize("alg_cls,alg_kwargs,model,dataset", get_algorithm_parametrization())
def test_get_algorithm_parametrization(
    alg_cls: Callable[..., Algorithm],
    alg_kwargs: Dict[str, Any],
    model: ComposerModel,
    dataset: Dataset,
):
    alg_instance = alg_cls(**alg_kwargs)
    assert isinstance(alg_instance, Algorithm)
    assert isinstance(model, ComposerModel)
    assert isinstance(dataset, Dataset)
