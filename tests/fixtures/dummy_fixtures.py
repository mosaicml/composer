# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable, Tuple

import pytest
import torch
import torch.utils.data
from torch.optim import Optimizer

from composer.core import Precision, State
from composer.core.types import PyTorchScheduler
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.datasets.dataset_hparams_registry import dataset_registry
from composer.models import ModelHparams
from composer.optim import ExponentialScheduler
from composer.optim.optimizer_hparams_registry import AdamHparams
from composer.trainer.trainer_hparams import TrainerHparams, model_registry
from tests.common import RandomClassificationDatasetHparams, SimpleModel, SimpleModelHparams


@pytest.fixture
def dummy_in_shape() -> Tuple[int, ...]:
    return 1, 5, 5


@pytest.fixture
def dummy_num_classes() -> int:
    return 2


@pytest.fixture()
def dummy_train_batch_size() -> int:
    return 16


@pytest.fixture()
def dummy_val_batch_size() -> int:
    return 32


@pytest.fixture()
def dummy_train_n_samples() -> int:
    return 1000


@pytest.fixture
def dummy_model_hparams(dummy_in_shape: Tuple[int, ...], dummy_num_classes: int) -> SimpleModelHparams:
    model_registry['simple'] = SimpleModelHparams
    return SimpleModelHparams(num_features=dummy_in_shape[0], num_classes=dummy_num_classes)


@pytest.fixture
def dummy_model(dummy_in_shape: Tuple[int, ...], dummy_num_classes: int) -> SimpleModel:
    return SimpleModel(num_features=dummy_in_shape[0], num_classes=dummy_num_classes)


@pytest.fixture
def dummy_train_dataset_hparams(dummy_model: SimpleModel, dummy_in_shape: Tuple[int]) -> DatasetHparams:
    dataset_registry['random_classification'] = RandomClassificationDatasetHparams
    assert dummy_model.num_classes is not None
    return RandomClassificationDatasetHparams(
        use_synthetic=True,
        drop_last=True,
        shuffle=False,
        num_classes=dummy_model.num_classes,
        data_shape=list(dummy_in_shape),
    )


@pytest.fixture
def dummy_val_dataset_hparams(dummy_model: SimpleModel, dummy_in_shape: Tuple[int]) -> DatasetHparams:
    dataset_registry['random_classification'] = RandomClassificationDatasetHparams
    assert dummy_model.num_classes is not None
    return RandomClassificationDatasetHparams(
        use_synthetic=True,
        drop_last=False,
        shuffle=False,
        num_classes=dummy_model.num_classes,
        data_shape=list(dummy_in_shape),
    )


@pytest.fixture
def dummy_optimizer(dummy_model: SimpleModel):
    return torch.optim.SGD(dummy_model.parameters(), lr=0.001)


@pytest.fixture
def dummy_scheduler(dummy_optimizer: Optimizer):
    return torch.optim.lr_scheduler.LambdaLR(dummy_optimizer, lambda _: 1.0)


@pytest.fixture()
def dummy_state(
    dummy_model: SimpleModel,
    dummy_train_dataloader: Iterable,
    dummy_optimizer: Optimizer,
    dummy_scheduler: PyTorchScheduler,
    rank_zero_seed: int,
    request: pytest.FixtureRequest,
) -> State:
    if request.node.get_closest_marker('gpu') is not None:
        # If using `dummy_state`, then not using the trainer, so move the model to the correct device
        dummy_model = dummy_model.cuda()
    state = State(
        model=dummy_model,
        run_name='dummy_run_name',
        precision=Precision.FP32,
        grad_accum=1,
        rank_zero_seed=rank_zero_seed,
        optimizers=dummy_optimizer,
        max_duration='10ep',
    )
    state.schedulers = dummy_scheduler
    state.set_dataloader(dummy_train_dataloader, 'train')

    return state


@pytest.fixture
def dummy_dataloader_hparams() -> DataLoaderHparams:
    return DataLoaderHparams(
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        timeout=0.0,
    )


@pytest.fixture
def dummy_train_dataloader(
    dummy_train_dataset_hparams: DatasetHparams,
    dummy_train_batch_size: int,
    dummy_dataloader_hparams: DataLoaderHparams,
):
    return dummy_train_dataset_hparams.initialize_object(dummy_train_batch_size, dummy_dataloader_hparams)


@pytest.fixture
def dummy_val_dataloader(
    dummy_train_dataset_hparams: DatasetHparams,
    dummy_val_batch_size: int,
    dummy_dataloader_hparams: DataLoaderHparams,
):
    return dummy_train_dataset_hparams.initialize_object(dummy_val_batch_size, dummy_dataloader_hparams)


@pytest.fixture
def composer_trainer_hparams(
    dummy_model_hparams: ModelHparams,
    dummy_train_dataset_hparams: DatasetHparams,
    dummy_val_dataset_hparams: DatasetHparams,
    dummy_train_batch_size: int,
    dummy_val_batch_size: int,
    rank_zero_seed: int,
) -> TrainerHparams:
    return TrainerHparams(
        algorithms=[],
        optimizers=AdamHparams(),
        schedulers=[ExponentialScheduler(gamma=0.9)],
        max_duration='2ep',
        precision=Precision.FP32,
        train_batch_size=dummy_train_batch_size,
        eval_batch_size=dummy_val_batch_size,
        seed=rank_zero_seed,
        dataloader=DataLoaderHparams(
            num_workers=0,
            prefetch_factor=2,
            persistent_workers=False,
            pin_memory=False,
            timeout=0.0,
        ),
        model=dummy_model_hparams,
        val_dataset=dummy_val_dataset_hparams,
        train_dataset=dummy_train_dataset_hparams,
        grad_accum=1,
        train_subset_num_batches=3,
        eval_subset_num_batches=3,
    )
