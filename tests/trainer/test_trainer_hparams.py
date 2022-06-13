# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Callable, Type

import pytest

from composer.datasets.dataset_hparams import DataLoaderHparams
from composer.trainer import Trainer
from composer.trainer.trainer_hparams import (EvalHparams, EvalKwargs, ExperimentHparams, FitHparams, FitKwargs,
                                              TrainerHparams)
from tests.common import SimpleModelHparams
from tests.common.datasets import RandomClassificationDatasetHparams

if TYPE_CHECKING:
    from typing import TypedDict


@pytest.mark.parametrize('method,typeddict_cls', [[Trainer.fit, FitKwargs], [Trainer.eval, EvalKwargs]])
def test_kwargs_match_signature(method: Callable, typeddict_cls: Type[TypedDict]):
    assert typing.get_type_hints(method) == typing.get_type_hints(typeddict_cls)


def test_experiment_hparams_initialize():
    """Test that the ExperimentHparams class initializes."""
    experiment_hparams = ExperimentHparams(
        trainer=TrainerHparams(
            model=SimpleModelHparams(),
            dataloader=DataLoaderHparams(
                num_workers=0,
                persistent_workers=False,
                pin_memory=False,
            ),
            max_duration=1,
        ),
        fits=[
            FitHparams(
                train_dataset=RandomClassificationDatasetHparams(),
                train_batch_size=1,
                train_subset_num_batches=1,
            ),
        ],
        evals=[EvalHparams(
            dataset=RandomClassificationDatasetHparams(),
            batch_size=1,
            subset_num_batches=1,
        )])
    trainer, fits, evals = experiment_hparams.initialize_object()

    for fit_kwargs in fits:
        trainer.fit(**fit_kwargs)

    for eval_kwargs in evals:
        trainer.eval(**eval_kwargs)
