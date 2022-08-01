# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import collections.abc
import os
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import pytest
import torch
import torch.distributed
import torch.utils.data
import yahp as hp

import composer.core.types as types
from composer import Callback, Event
from composer.core import State
from composer.core.data_spec import DataSpec
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.datasets.synthetic_hparams import SyntheticHparamsMixin
from composer.loggers import Logger
from composer.models.model_hparams import ModelHparams
from composer.trainer.trainer import Trainer
from composer.utils import dist
from tests.common import SimpleModel


def get_file_path(*, is_train: bool, tmp_path: pathlib.Path) -> str:
    train_str = 'train' if is_train else 'val'
    file_path = os.path.join(tmp_path, f'{train_str}_num_accesses')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return file_path


def get_batch_file_path(*, epoch: int, is_train: bool, tmp_path: pathlib.Path) -> str:
    train_str = 'train' if is_train else 'val'
    file_path = os.path.join(tmp_path, f'{train_str}-epoch-{epoch}-batch0.pt')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return file_path


class TrackedDataset(types.Dataset):
    """TrackedDataset atomically writes a file every time a record is accessed.

    It is thread-safe and subprocess-safe, and is useful to measure how many times a sample is accessed. Because of
    atomic file writes, it is slow and should not be used in any performance measurements.
    """

    def __init__(self, is_train: bool, synthetic_dataset: SyntheticBatchPairDataset, tmp_path: pathlib.Path):
        self.dataset = synthetic_dataset
        self.is_train = is_train
        self.tmp_path = tmp_path
        self.counter = 0

    def __getitem__(self, idx: int):
        self.counter += 1
        with open(get_file_path(tmp_path=self.tmp_path, is_train=self.is_train), 'w+') as f:
            f.write(str(self.counter))
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


@dataclass
class TrackedDatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    num_classes: Optional[int] = hp.optional('num_classes', default=None)
    data_shape: Optional[List[int]] = hp.optional('data_shape', default=None)
    tmp_path: Optional[str] = hp.optional('tmp_path', default=None)
    is_train: bool = hp.optional('Whether to load the training data (the default) or validation data.', default=True)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams):
        assert self.num_classes is not None
        assert self.data_shape is not None
        assert self.tmp_path is not None
        synthetic_dataset = SyntheticBatchPairDataset(
            num_unique_samples_to_create=self.synthetic_num_unique_samples,
            total_dataset_size=10_000,
            data_shape=self.data_shape,
            num_classes=self.num_classes,
        )
        drop_last = False
        tracked_dataset = TrackedDataset(
            tmp_path=pathlib.Path(self.tmp_path),
            is_train=self.is_train,
            synthetic_dataset=synthetic_dataset,
        )
        sampler = dist.get_sampler(tracked_dataset, drop_last=drop_last, shuffle=True)
        return dataloader_hparams.initialize_object(
            dataset=tracked_dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
        )


class CheckBatch0(Callback):

    def __init__(self, tmp_path: pathlib.Path):
        self.tmp_path = tmp_path

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        if event in (Event.BEFORE_FORWARD, Event.EVAL_BEFORE_FORWARD):
            filepath = get_batch_file_path(
                epoch=int(state.timestamp.epoch),
                is_train=state.model.training,
                tmp_path=self.tmp_path,
            )
            if os.path.exists(filepath):
                return
            last_input, last_target = state.batch
            torch.save(
                {
                    'last_input': last_input,
                    'last_target': last_target,
                },
                filepath,
            )


@pytest.mark.parametrize('device,deepspeed', [
    pytest.param('cpu', False, id='cpu'),
    pytest.param('gpu', False, id='gpu', marks=pytest.mark.gpu),
    pytest.param('gpu', True, id='deepspeed', marks=pytest.mark.gpu),
])
@pytest.mark.parametrize('world_size', [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
def test_ddp(device: str, world_size: int, dummy_model_hparams: ModelHparams, deepspeed: bool,
             tmp_path: pathlib.Path) -> None:
    """test strategy for ddp: 1) Train a dummy model on two gps, for two epochs, using the tracked dataset. 2) The
    tracked dataset should record two -- and only two -- accesses for each sample -- one for each epoch If each sample
    is accessed more than this number of times, then the distributed sampler isn't working properly If each sample is
    accessed less than this number of times, then either the sample pool size isn't a multiple of the batch size (and
    samples are getting dropped), or not all processes are working 3) We use a callback to save the (x, y) for the first
    batch in each epoch on each process.

     ({train, eval} * {epoch 1, epoch 2} * {ddp 1, ddp2})
    We assert that each of these tensors are different to ensure that 1) random seeding works properly,
    and 2) each ddp process is indeed getting different data.
    """

    dummy_model_hparams.num_classes = 100
    model = dummy_model_hparams.initialize_object()
    assert isinstance(model, SimpleModel)

    dataloader_hparams = DataLoaderHparams(
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        timeout=0.0,
    )

    train_batch_size = 10
    train_dataloader_batch_size = train_batch_size // dist.get_world_size()
    train_subset_num_batches = 3
    train_dataset_hparams = TrackedDatasetHparams(
        synthetic_num_unique_samples=train_batch_size * train_subset_num_batches,
        is_train=True,
        data_shape=[model.num_features, 5, 5],
        num_classes=model.num_classes,
        tmp_path=str(tmp_path),
    )
    train_dataloader = train_dataset_hparams.initialize_object(train_dataloader_batch_size, dataloader_hparams)
    eval_batch_size = 10
    eval_subset_num_batches = 3
    val_dataset_hparams = TrackedDatasetHparams(
        synthetic_num_unique_samples=eval_batch_size * eval_subset_num_batches,
        is_train=False,
        data_shape=[model.num_features, 5, 5],
        num_classes=model.num_classes,
        tmp_path=str(tmp_path),
    )
    eval_dataloader_batch_size = eval_batch_size // dist.get_world_size()
    val_dataloader = val_dataset_hparams.initialize_object(eval_dataloader_batch_size, dataloader_hparams)
    max_epochs = 2
    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      eval_dataloader=val_dataloader,
                      device=device,
                      max_duration=f'{max_epochs}ep',
                      eval_interval='1ep',
                      eval_subset_num_batches=eval_subset_num_batches,
                      train_subset_num_batches=train_subset_num_batches,
                      deepspeed_config={} if deepspeed else None,
                      callbacks=[CheckBatch0(tmp_path)])

    for evaluator in trainer.state.evaluators:
        assert isinstance(evaluator.dataloader, DataSpec)
        assert isinstance(evaluator.dataloader.dataloader, collections.abc.Sized)
    trainer.fit()

    expected_train_num_loads = max_epochs * train_batch_size * train_subset_num_batches
    #expected_val_num_loads = max_epochs * hparams.eval_batch_size * hparams.eval_subset_num_batches
    expected_val_num_loads = 0
    for evaluator in trainer.state.evaluators:
        expected_val_num_loads += max_epochs * eval_batch_size * eval_subset_num_batches

    # adding hparams.eval_batch_size to account for the extra spin of the evaluator dataloaders
    # that is called to create a deterministic ordering for the sampler
    for evaluator in trainer.state.evaluators:
        expected_val_num_loads += eval_batch_size

    actual_train_num_loads = 0
    actual_val_num_loads = 0

    rank_to_tmp_path = [pathlib.Path(x) for x in dist.all_gather_object(str(tmp_path))]

    for rank_tmp_path in rank_to_tmp_path:
        with open(get_file_path(is_train=True, tmp_path=rank_tmp_path), 'r') as f:
            actual_train_num_loads += int(f.read())
        with open(get_file_path(is_train=False, tmp_path=rank_tmp_path), 'r') as f:
            actual_val_num_loads += int(f.read())
    assert actual_train_num_loads == expected_train_num_loads, f'actual_train_num_loads({actual_train_num_loads}) != expected_train_num_loads({expected_train_num_loads})'
    assert actual_val_num_loads == expected_val_num_loads, f'actual_val_num_loads({actual_val_num_loads}) != expected_val_num_loads({expected_val_num_loads})'

    is_train_to_pickles: Dict[bool, List[Dict[str, torch.Tensor]]] = {True: [], False: []}

    if deepspeed:
        # it is not possible to save individual batches when using deepspeed
        return

    for epoch in range(max_epochs):
        for is_train in (True, False):
            real_epoch = epoch if is_train else epoch + 1  # validation is 1 ahead of training
            data: Dict[str, torch.Tensor] = torch.load(
                get_batch_file_path(
                    epoch=real_epoch,
                    is_train=is_train,
                    tmp_path=tmp_path,
                ),
                map_location='cpu',
            )
            for pickle in is_train_to_pickles[is_train]:
                assert not torch.all(
                    data['last_input'] == pickle['last_input']
                ), f'inputs are the same for is_train={is_train}, epoch={epoch}, rank={dist.get_global_rank()}'
                assert not torch.all(
                    data['last_target'] == pickle['last_target']
                ), f'targets are the same for is_train={is_train}, epoch={epoch}, rank={dist.get_global_rank()}'
            is_train_to_pickles[is_train].append(data)
