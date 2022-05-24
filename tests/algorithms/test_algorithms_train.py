# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict

import pytest
from torch.utils.data import DataLoader, Dataset

from composer import Algorithm, ComposerModel, Trainer
from tests.algorithms.algorithm_settings import get_algorithm_parametrization
from tests.common import device


class TestTrainerAlgorithms:

    @pytest.mark.timeout(5)
    @device('gpu')
    @pytest.mark.parametrize("alg_cls,alg_kwargs,model,dataset", get_algorithm_parametrization())
    def test_algorithm_trains(
        self,
        device: str,
        alg_cls: Callable[..., Algorithm],
        alg_kwargs: Dict[str, Any],
        model: ComposerModel,
        dataset: Dataset,
    ):
        trainer = Trainer(
            model=model,
            train_dataloader=DataLoader(dataset=dataset, batch_size=4),
            max_duration='2ep',
            device=device,
            algorithms=alg_cls(**alg_kwargs),
        )
        trainer.fit()

        # fit again for another epoch
        trainer.fit(duration='1ep')
