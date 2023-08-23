# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
import os
import pathlib
import uuid
from unittest.mock import MagicMock

import pytest
from torch.utils.data import DataLoader

from composer import Trainer
from composer._version import __version__
from composer.loggers import NeptuneLogger
from composer.utils import dist
from tests.common import RandomImageDataset, SimpleConvModel
from tests.common.markers import device


@pytest.fixture
def test_neptune_logger() -> NeptuneLogger:
    neptune_project = 'test_project'
    neptune_api_token = 'test_token'

    neptune_logger = NeptuneLogger(
        project=neptune_project,
        api_token=neptune_api_token,
        rank_zero_only=False,
        mode='debug',
        log_artifacts=True,
    )

    return neptune_logger


def test_neptune_init(test_neptune_logger):
    mock_state = MagicMock()
    mock_state.run_name = 'dummy-run-name'  # should appear in sys/tags

    test_neptune_logger.init(state=mock_state, logger=MagicMock())

    assert test_neptune_logger.neptune_run is not None

    assert test_neptune_logger.neptune_run[NeptuneLogger.INTEGRATION_VERSION_KEY].fetch() == __version__
    assert test_neptune_logger.neptune_run['sys/tags'].fetch() == {'rank0', 'dummy-run-name'}


@device('cpu')
def test_neptune_logging(device, test_neptune_logger):

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


def test_upload_and_download_file(test_neptune_logger, tmp_path, dummy_state):
    neptune_artifact_name = 'test-neptune-artifact-' + str(uuid.uuid4())
    tmp_paths = dist.all_gather_object(os.path.abspath(tmp_path))
    save_folder = pathlib.Path(tmp_paths[0])
    file_content = 'hello from Neptune!'

    dummy_neptune_artifact_path = save_folder / 'neptune_artifact.txt'
    if dist.get_global_rank() == 0:
        with open(dummy_neptune_artifact_path, 'w+') as f:
            f.write(file_content)

    test_neptune_logger.upload_file(state=dummy_state,
                                    file_path=dummy_neptune_artifact_path,
                                    remote_file_name=neptune_artifact_name)

    dist.barrier()

    assert test_neptune_logger.neptune_run.exists(f'{test_neptune_logger._base_namespace}/{neptune_artifact_name}')

    dst_path = save_folder / 'neptune_artifact'

    test_neptune_logger.download_file(
        remote_file_name=neptune_artifact_name,
        destination=str(dst_path),
    )

    assert dst_path.exists()

    with open(str(dst_path), 'r') as fp:
        assert fp.read() == file_content
