# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import torch.utils.data
from _pytest.monkeypatch import MonkeyPatch
from tqdm import auto

from composer.core.time import Time, TimeUnit
from composer.trainer.trainer import Trainer
from composer.utils import dist
from tests.common import RandomClassificationDataset, SimpleModel


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

    model = SimpleModel()

    monkeypatch.setattr(auto, 'tqdm', get_mock_tqdm)

    eval_interval = 1
    eval_subset_num_batches = 2
    batch_size = 10
    train_dataset = RandomClassificationDataset()
    eval_dataset = RandomClassificationDataset()

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

    trainer.fit()

    if dist.get_local_rank() != 0:
        return

    # either have #epoch pbars, or only have 1 train pbar
    if max_duration.unit == TimeUnit.EPOCH:
        assert len(mock_tqdms_train) == max_duration.value
    else:
        assert len(mock_tqdms_train) == 1

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
