# Copyright 2021 MosaicML. All Rights Reserved.

from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from tqdm import auto

from composer.loggers import ProgressBarLoggerHparams
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils import dist


@pytest.mark.parametrize("world_size", [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
def test_progress_bar_logger(composer_trainer_hparams: TrainerHparams, monkeypatch: MonkeyPatch, world_size: int):
    is_train_to_mock_tqdms = {
        True: [],
        False: [],
    }

    def get_mock_tqdm(position: int, *args: object, **kwargs: object):
        del args, kwargs  # unused
        is_train = position == 0
        mock_tqdm = MagicMock()
        is_train_to_mock_tqdms[is_train].append(mock_tqdm)
        return mock_tqdm

    monkeypatch.setattr(auto, "tqdm", get_mock_tqdm)

    max_epochs = 2
    composer_trainer_hparams.max_duration = f"{max_epochs}ep"
    composer_trainer_hparams.loggers = [ProgressBarLoggerHparams()]
    trainer = composer_trainer_hparams.initialize_object()
    trainer.fit()
    if dist.get_global_rank() == 1:
        return
    assert len(is_train_to_mock_tqdms[True]) == max_epochs
    assert composer_trainer_hparams.validate_every_n_batches < 0
    assert len(is_train_to_mock_tqdms[False]) == composer_trainer_hparams.validate_every_n_epochs * max_epochs
    for mock_tqdm in is_train_to_mock_tqdms[True]:
        assert mock_tqdm.update.call_count == trainer.state.steps_per_epoch
        mock_tqdm.close.assert_called_once()
    for mock_tqdm in is_train_to_mock_tqdms[False]:
        assert mock_tqdm.update.call_count == trainer._eval_subset_num_batches
        mock_tqdm.close.assert_called_once()
