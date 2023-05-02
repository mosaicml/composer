# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

import pytest
import torch
import torch.distributed
from packaging import version
from torch.utils.data import DataLoader

import composer.core.types as types
from composer import Callback, Event
from composer.core import State
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.loggers import Logger
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


@pytest.mark.parametrize(
    'device,deepspeed,fsdp',
    [
        pytest.param('cpu', False, False, id='cpu'),
        pytest.param('gpu', False, False, id='gpu', marks=pytest.mark.gpu),
        # TODO: Remove filterwarnings after FSDP removes deprecated code
        pytest.param('gpu', True, False, id='deepspeed', marks=pytest.mark.gpu),
        pytest.param('gpu',
                     False,
                     True,
                     id='fsdp',
                     marks=[
                         pytest.mark.gpu,
                         pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                                            reason='requires PyTorch 1.13 or higher'),
                         pytest.mark.filterwarnings('ignore::UserWarning'),
                     ]),
    ])
@pytest.mark.parametrize('world_size', [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
def test_ddp(device: str, world_size: int, deepspeed: bool, fsdp: bool, tmp_path: pathlib.Path) -> None:
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

    model = SimpleModel(num_classes=100)

    train_batch_size = 10
    train_subset_num_batches = 3

    synthetic_dataset = SyntheticBatchPairDataset(
        num_unique_samples_to_create=train_batch_size * train_subset_num_batches,
        total_dataset_size=10_000,
        data_shape=(model.num_features, 5, 5),
        num_classes=model.num_classes,
    )
    train_dataset = TrackedDataset(
        synthetic_dataset=synthetic_dataset,
        is_train=True,
        tmp_path=tmp_path,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        timeout=0.0,
        batch_size=train_batch_size // dist.get_world_size(),
        sampler=dist.get_sampler(
            train_dataset,
            drop_last=False,
            shuffle=True,
        ),
    )

    eval_batch_size = 10
    eval_subset_num_batches = 3

    eval_dataset = SyntheticBatchPairDataset(
        num_unique_samples_to_create=eval_batch_size * eval_subset_num_batches,
        total_dataset_size=10_000,
        data_shape=(model.num_features, 5, 5),
        num_classes=model.num_classes,
    )
    eval_dataset = TrackedDataset(
        synthetic_dataset=eval_dataset,
        is_train=False,
        tmp_path=tmp_path,
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=eval_batch_size // dist.get_world_size(),
        sampler=dist.get_sampler(
            eval_dataset,
            drop_last=False,
            shuffle=True,
        ),
    )

    fsdp_config = None
    if fsdp:
        fsdp_config = {
            'sharding_strategy': 'FULL_SHARD',
            'min_params': 1e8,
            'cpu_offload': False,
            'mixed_precision': 'PURE',
            'backward_prefetch': 'BACKWARD_PRE',
            'activation_checkpointing': False,
            'activation_cpu_offload': False,
            'verbose': False
        }

    max_epochs = 2
    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      eval_dataloader=eval_dataloader,
                      device=device,
                      max_duration=f'{max_epochs}ep',
                      eval_interval='1ep',
                      eval_subset_num_batches=eval_subset_num_batches,
                      train_subset_num_batches=train_subset_num_batches,
                      deepspeed_config={} if deepspeed else None,
                      fsdp_config=fsdp_config,
                      callbacks=[CheckBatch0(tmp_path)])

    trainer.fit()

    expected_train_samples = max_epochs * train_batch_size * train_subset_num_batches

    expected_val_samples = max_epochs * eval_batch_size * eval_subset_num_batches
    # account for extra spin to create deterministic ordering
    expected_val_samples += eval_batch_size

    actual_train_samples = _read_tracked_results(tmp_path, is_train=True)
    actual_val_samples = _read_tracked_results(tmp_path, is_train=False)

    assert expected_train_samples == actual_train_samples
    assert expected_val_samples == actual_val_samples

    if not deepspeed:
        _assert_inputs_different(tmp_path, max_epochs, is_train=True)
        _assert_inputs_different(tmp_path, max_epochs, is_train=False)


def _read_tracked_results(path, is_train):

    # get all paths across ranks
    paths = [pathlib.Path(p) for p in dist.all_gather_object(str(path))]

    counter = 0
    for p in paths:
        with open(get_file_path(is_train=is_train, tmp_path=p), 'r') as f:
            counter += int(f.read())
    return counter


def _assert_inputs_different(tmp_path, max_epochs, is_train):
    """Checks that each rank's dataloader input is different."""

    inputs = []
    targets = []
    for epoch in range(max_epochs):

        file_path = get_batch_file_path(
            epoch=epoch if is_train else epoch + 1,  # val is 1 ahead
            is_train=is_train,
            tmp_path=tmp_path,
        )
        state_dict = torch.load(file_path, map_location='cpu')

        for input in inputs:
            if torch.allclose(state_dict['last_input'], input):
                raise ValueError(f'Tensors equal for epoch {epoch}, rank {dist.get_global_rank()}')

        for target in targets:
            if torch.allclose(state_dict['last_target'], target):
                raise ValueError(f'Tensors equal for epoch {epoch}, rank {dist.get_global_rank()}')

        inputs.append(state_dict['last_input'])
        targets.append(state_dict['last_target'])
