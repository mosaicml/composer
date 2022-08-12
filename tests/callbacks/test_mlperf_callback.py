# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# ignore third-party missing imports due to the mlperf logger not pip-installable
# pyright: reportMissingImports=none

import logging
from unittest.mock import Mock

import numpy as np
import pytest
from torch.utils.data import DataLoader

from composer import State, Trainer
from composer.callbacks import MLPerfCallback
from composer.utils import dist
from tests.common import RandomClassificationDataset, SimpleModel
from tests.common.markers import device, world_size


def rank_zero() -> bool:
    return dist.get_global_rank() == 0


@pytest.fixture(autouse=True)
def importor_skip_mlperf_logging():
    pytest.importorskip('mlperf_logging')


class MockMLLogger:
    """Mocks the MLPerf Logger interface."""

    def __init__(self) -> None:
        self.logs = []
        self.logger = Mock()

    def event(self, key, metadata, value=None):
        self.logs.append({'key': key, 'value': value, 'metadata': metadata})


class TestMLPerfCallbackEvents:

    @pytest.fixture
    def mlperf_callback(self, monkeypatch, tmp_path) -> MLPerfCallback:
        """Returns a callback with the MockMLLogger patched."""
        callback = MLPerfCallback(tmp_path, 0)
        monkeypatch.setattr(callback, 'mllogger', MockMLLogger())
        return callback

    @pytest.fixture
    def mock_state(self):
        """Mocks a state at epoch 1 with Accuracy 0.99."""
        current_metrics = {'eval': {'Accuracy': 0.99}}

        state = Mock()
        state.current_metrics = current_metrics
        state.timestamp.epoch.value = 1

        return state

    def test_eval_start(self, mlperf_callback, mock_state):
        mlperf_callback.eval_start(mock_state, Mock())

        if not rank_zero():
            assert mlperf_callback.mllogger.logs == []
            return

        assert mlperf_callback.mllogger.logs == [{'key': 'eval_start', 'value': None, 'metadata': {'epoch_num': 1}}]

    def test_eval_end(self, mlperf_callback, mock_state):
        mlperf_callback.eval_end(mock_state, Mock())

        if not rank_zero():
            assert mlperf_callback.success == False
            assert mlperf_callback.mllogger.logs == []
            return

        assert mlperf_callback.success == True
        assert mlperf_callback.mllogger.logs[-1] == {
            'key': 'run_stop',
            'value': None,
            'metadata': {
                'status': 'success'
            }
        }


@world_size(1, 2)
@device('cpu', 'gpu')
class TestWithMLPerfChecker:
    """Ensures that the logs created by the MLPerfCallback pass the official package checker."""

    def test_mlperf_callback_passes(self, tmp_path, monkeypatch, world_size, device):

        def mock_accuracy(self, state: State):
            if state.timestamp.epoch >= 2:
                return 0.99
            else:
                return 0.01

        monkeypatch.setattr(MLPerfCallback, '_get_accuracy', mock_accuracy)

        self.generate_submission(tmp_path, device)

        if rank_zero():
            self.run_mlperf_checker(tmp_path, monkeypatch)

    def test_mlperf_callback_fails(self, tmp_path, monkeypatch, world_size, device):

        def mock_accuracy(self, state: State):
            return 0.01

        monkeypatch.setattr(MLPerfCallback, '_get_accuracy', mock_accuracy)

        self.generate_submission(tmp_path, device)
        with pytest.raises(ValueError, match='MLPerf checker failed'):
            self.run_mlperf_checker(tmp_path, monkeypatch)

    def generate_submission(self, directory, device):
        """Generates submission files by training the benchark n=5 times."""

        for run in range(5):

            mlperf_callback = MLPerfCallback(
                root_folder=directory,
                index=run,
                cache_clear_cmd='sleep 0.1',
            )

            trainer = Trainer(
                model=SimpleModel(),
                train_dataloader=DataLoader(
                    dataset=RandomClassificationDataset(),
                    batch_size=4,
                    shuffle=False,
                ),
                eval_dataloader=DataLoader(
                    dataset=RandomClassificationDataset(),
                    shuffle=False,
                ),
                max_duration='3ep',
                deterministic_mode=True,
                progress_bar=False,
                log_to_console=False,
                loggers=[],
                device=device,
                callbacks=[mlperf_callback],
                seed=np.random.randint(low=2048),
            )

            trainer.fit()

    def run_mlperf_checker(self, directory, monkeypatch):
        """Runs the MLPerf package checker and fails on any errors."""

        # monkeypatch the logging so that logging.error raises Exception
        def fail_on_error(msg, *args, **kwargs):
            print(msg.format(*args))
            raise ValueError('MLPerf checker failed, see logs.')

        monkeypatch.setattr(logging, 'error', fail_on_error)

        from mlperf_logging.package_checker.package_checker import check_training_package

        check_training_package(
            folder=directory,
            usage='training',
            ruleset='1.1.0',
            werror=True,
            quiet=False,
            rcp_bypass=False,
            rcp_bert_train_samples=False,
            log_output='package_checker.log',
        )
