# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Type

import pytest

from composer import Algorithm, Trainer
from composer.algorithms.layer_freezing.layer_freezing import LayerFreezing
from tests.algorithms.algorithm_settings import get_alg_dataloader, get_alg_kwargs, get_alg_model, get_algs_with_marks


@pytest.mark.gpu
@pytest.mark.parametrize('alg_cls', get_algs_with_marks())
def test_algorithm_trains(alg_cls: Type[Algorithm]):
    alg_kwargs = get_alg_kwargs(alg_cls)
    model = get_alg_model(alg_cls)
    dataloader = get_alg_dataloader(alg_cls)
    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        max_duration='2ep',
        algorithms=alg_cls(**alg_kwargs),
    )
    trainer.fit()

    if alg_cls is LayerFreezing:
        pytest.xfail(('Layer freezing is incompatible with a second call to .fit() '
                      'since all layers are frozen, and it does not unfreeze layers.'))

    # fit again for another epoch
    trainer.fit(duration='1ep')
