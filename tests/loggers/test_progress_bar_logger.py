# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import torch.utils.data
from _pytest.monkeypatch import MonkeyPatch
from torchmetrics import Accuracy
from tqdm import auto

from composer.core.evaluator import Evaluator
from composer.core.time import Time, TimeUnit
from composer.loggers.progress_bar_logger import ProgressBarLogger
from composer.trainer.trainer import Trainer
from composer.utils import dist
from tests.common import RandomClassificationDataset, SimpleModel


def run_trainer_with_mock_pbar(monkeypatch: MonkeyPatch, trainer: Trainer):
    mock_tqdms_train = []
    mock_tqdms_eval = []

    def get_mock_tqdm(bar_format: str, *args: object, **kwargs: object):
        del args, kwargs  # unused
        mock_tqdm = MagicMock()
        mock_tqdm.n = 0

        # store for testing later
        if 'train' in bar_format:
            mock_tqdms_train.append(mock_tqdm)
        if 'eval' in bar_format:
            mock_tqdms_eval.append(mock_tqdm)

        return mock_tqdm

    monkeypatch.setattr(auto, 'tqdm', get_mock_tqdm)

    trainer.fit()

    return mock_tqdms_train, mock_tqdms_eval


@pytest.mark.parametrize('world_size', [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
@pytest.mark.parametrize(
    'max_duration',
    [Time.from_timestring('2ep'),
     Time.from_timestring('100sp'),
     Time.from_timestring('5ba')],
)
def test_progress_bar_logger(max_duration: Time[int], monkeypatch: MonkeyPatch, world_size: int):

    eval_interval = 1
    eval_subset_num_batches = 2
    batch_size = 10
    train_dataset = RandomClassificationDataset()
    eval_dataset = RandomClassificationDataset()
    model = SimpleModel()

    trainer = Trainer(
        model=model,
        max_duration=max_duration,
        eval_interval=eval_interval,
        progress_bar=True,
        compute_training_metrics=True,
        train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size),
        eval_dataloader=torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size),
        eval_subset_num_batches=eval_subset_num_batches,
    )

    mock_tqdms_train, mock_tqdms_eval = run_trainer_with_mock_pbar(monkeypatch, trainer)

    if dist.get_local_rank() != 0:
        return

    # test train pbar
    if max_duration.unit == TimeUnit.EPOCH:
        for mt in mock_tqdms_train:
            assert trainer.state.dataloader_len is not None
            assert mt.update.call_count == int(trainer.state.dataloader_len)
    elif max_duration.unit == TimeUnit.BATCH:
        for mt in mock_tqdms_train:
            assert mt.update.call_count == max_duration.value
    elif max_duration.unit == TimeUnit.SAMPLE:
        for mt in mock_tqdms_train:
            assert mt.update.call_count == max_duration.value // batch_size / world_size

    # test eval pbar
    for mt in mock_tqdms_eval:
        assert mt.update.call_count == eval_subset_num_batches


@pytest.mark.parametrize('world_size', [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
@pytest.mark.parametrize(
    'max_duration',
    [Time.from_timestring('2ep'),
     Time.from_timestring('100sp'),
     Time.from_timestring('5ba')],
)
def test_progress_bar_dataloader_label(max_duration: Time[int], monkeypatch: MonkeyPatch, world_size: int):
    model = SimpleModel()

    eval_interval = 1
    eval_subset_num_batches = 2
    batch_size = 10

    dataloader1 = torch.utils.data.DataLoader(RandomClassificationDataset(), batch_size=batch_size)
    dataloader2 = torch.utils.data.DataLoader(RandomClassificationDataset(), batch_size=batch_size)

    evaluator1 = Evaluator(label='eval1',
                           metrics=Accuracy(),
                           dataloader=dataloader1,
                           eval_interval=eval_interval,
                           subset_num_batches=eval_subset_num_batches)
    evaluator2 = Evaluator(label='eval2',
                           metrics=Accuracy(),
                           dataloader=dataloader2,
                           eval_interval=eval_interval,
                           subset_num_batches=eval_subset_num_batches)

    trainer = Trainer(
        model=model,
        max_duration=max_duration,
        eval_interval=eval_interval,
        eval_dataloader=[evaluator1, evaluator2],
        compute_training_metrics=True,
        train_dataloader=torch.utils.data.DataLoader(RandomClassificationDataset(), batch_size=batch_size),
        eval_subset_num_batches=eval_subset_num_batches,
        loggers=[ProgressBarLogger(True, dataloader_label='eval2'),
                 ProgressBarLogger(True, dataloader_label='train')])

    mock_tqdms_train, mock_tqdms_eval = run_trainer_with_mock_pbar(monkeypatch, trainer)

    if dist.get_local_rank() != 0:
        return

    # test train pbar
    if max_duration.unit == TimeUnit.EPOCH:
        for mt in mock_tqdms_train:
            assert trainer.state.dataloader_len is not None
            assert mt.update.call_count == int(trainer.state.dataloader_len)
    elif max_duration.unit == TimeUnit.BATCH:
        for mt in mock_tqdms_train:
            assert mt.update.call_count == max_duration.value
    elif max_duration.unit == TimeUnit.SAMPLE:
        for mt in mock_tqdms_train:
            assert mt.update.call_count == max_duration.value // batch_size / world_size

    # test eval pbar
    for mt in mock_tqdms_eval:
        assert mt.update.call_count == eval_subset_num_batches
