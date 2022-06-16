# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Type

import pytest
from torch.utils.data import DataLoader

from composer.core.algorithm import Algorithm
from composer.models.base import ComposerModel
from tests.algorithms.algorithm_settings import get_alg_dataloader, get_alg_kwargs, get_alg_model, get_algs_with_marks


@pytest.mark.parametrize('alg_cls', get_algs_with_marks())
def test_algorithm_settings(alg_cls: Type[Algorithm]):
    alg_kwargs = get_alg_kwargs(alg_cls)
    alg_instance = alg_cls(**alg_kwargs)
    dataloader = get_alg_dataloader(alg_cls)
    model = get_alg_model(alg_cls)

    assert isinstance(alg_instance, Algorithm)
    assert isinstance(model, ComposerModel)
    assert isinstance(dataloader, DataLoader)
