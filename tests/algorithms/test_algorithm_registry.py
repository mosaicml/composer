# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import Dataset

from composer.algorithms.algorithm_hparams import algorithm_registry
from composer.core.algorithm import Algorithm
from composer.models.base import ComposerModel
from tests.algorithms.algorithm_settings import get_settings


@pytest.mark.parametrize("name", algorithm_registry)
def test_algorithm_settings(name):
    if name in ('alibi', 'seq_length_warmup', 'factorize', 'no_op_model', 'scale_schedule'):
        pytest.skip()

    setting = get_settings(name)
    if setting is None:
        pytest.skip()

    assert isinstance(setting['algorithm'], Algorithm)
    assert isinstance(setting['model'], ComposerModel)
    assert isinstance(setting['dataset'], Dataset)
