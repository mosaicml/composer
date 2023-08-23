# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from torch.utils.data import DataLoader

from composer import Trainer
from composer._version import __version__
from composer.loggers import NeptuneLogger
from tests.common import RandomImageDataset, SimpleConvModel
from tests.common.markers import device


def test_neptune_init_specified(monkeypatch):
    mock_state = MagicMock()
    mock_state.run_name = 'dummy-run-name'  # should appear in sys/tags

    neptune_project = 'test_project'
    neptune_api_token = 'test_token'

    specified = NeptuneLogger(
        project=neptune_project,
        api_token=neptune_api_token,
        rank_zero_only=False,
        mode='debug',
    )

    specified.init(state=mock_state, logger=MagicMock())

    assert specified.neptune_run is not None

    assert specified.neptune_run[NeptuneLogger.INTEGRATION_VERSION_KEY].fetch() == __version__
    assert specified.neptune_run['sys/tags'].fetch() == {'rank0', 'dummy-run-name'}


@device('cpu')
def test_neptune_logging(device):
    neptune_project = 'test_project'
    neptune_api_token = 'test_token'

    test_neptune_logger = NeptuneLogger(
        project=neptune_project,
        api_token=neptune_api_token,
        rank_zero_only=False,
        mode='debug',
    )

    dataset_size = 64
    batch_size = 4
    num_batches = 4
    eval_interval = '1ba'

    trainer = Trainer(model=SimpleConvModel(),
                      loggers=test_neptune_logger,
                      train_dataloader=DataLoader(RandomImageDataset(size=dataset_size), batch_size),
                      eval_dataloader=DataLoader(RandomImageDataset(size=dataset_size), batch_size),
                      max_duration=f'{num_batches}ba',
                      eval_interval=eval_interval,
                      device=device)
    trainer.fit()

    assert test_neptune_logger.neptune_run is not None
    assert test_neptune_logger.base_handler is not None

    for metric_name in [
            'metrics/train/MulticlassAccuracy', 'metrics/eval/MulticlassAccuracy', 'metrics/eval/CrossEntropy',
            'loss/train/total'
    ]:
        path = f'{test_neptune_logger._base_namespace}/{test_neptune_logger.METRIC_NAMESPACE}/{metric_name}'
        assert test_neptune_logger.neptune_run.exists(path)

    for hyperparam_name in ['node_name', 'num_cpus_per_node', 'num_nodes', 'rank_zero_seed']:
        path = f'{test_neptune_logger._base_namespace}/{test_neptune_logger.HYPERPARAM_NAMESPACE}/{hyperparam_name}'
        assert test_neptune_logger.neptune_run.exists(path)

    assert test_neptune_logger.base_handler['hyperparameters/num_nodes'].fetch() == 1
