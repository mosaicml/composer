# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
import pathlib
import uuid
from typing import Type

import pytest
import torch
from torch.utils.data import DataLoader

from composer.core import Engine, Event
from composer.core.callback import Callback
from composer.core.state import State
from composer.loggers import InMemoryLogger
from composer.loggers.logger import Logger, LogLevel
from composer.loggers.wandb_logger import WandBLogger
from composer.trainer import Trainer
from composer.utils import dist, retry
from tests.callbacks.callback_settings import get_cb_kwargs, get_cbs_and_marks
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.mark.parametrize('callback_cls', get_cbs_and_marks(callbacks=True))
def test_logged_data_is_json_serializable(callback_cls: Type[Callback]):
    """Test that all logged data is json serializable, which is a requirement to use wandb."""
    pytest.importorskip('wandb', reason='wandb is optional')
    from wandb.sdk.data_types.base_types.wb_value import WBValue
    callback_kwargs = get_cb_kwargs(callback_cls)
    callback = callback_cls(**callback_kwargs)
    logger = InMemoryLogger()  # using an in memory logger to manually validate json serializability
    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=DataLoader(RandomClassificationDataset()),
        train_subset_num_batches=2,
        max_duration='1ep',
        callbacks=callback,
        loggers=logger,
        compute_training_metrics=True,
    )
    trainer.fit()

    for log_calls in logger.data.values():
        for timestamp, log_level, data in log_calls:
            del timestamp, log_level  # unused
            # manually filter out custom W&B data types and tensors, which are allowed, but cannot be json serialized
            if isinstance(data, (WBValue, torch.Tensor)):
                continue
            json.dumps(data)


@pytest.mark.world_size(2)
@pytest.mark.parametrize('rank_zero_only', [True, False])
@pytest.mark.remote
def test_wandb_artifacts(rank_zero_only: bool, tmp_path: pathlib.Path, dummy_state: State):
    """Test that artifacts logged on rank zero are accessible by all ranks."""
    pytest.importorskip('wandb', reason='wandb is optional')
    # Create the logger
    ctx = pytest.warns(
        UserWarning, match='`rank_zero_only` should be set to False.') if rank_zero_only else contextlib.nullcontext()
    with ctx:
        wandb_logger = WandBLogger(
            rank_zero_only=rank_zero_only,
            log_artifacts=True,
        )
    dummy_state.callbacks.append(wandb_logger)
    logger = Logger(dummy_state, [wandb_logger])
    engine = Engine(dummy_state, logger)
    engine.run_event(Event.INIT)

    # Distribute the artifact name from rank 0 to all ranks
    artifact_name = 'test-artifact-' + str(uuid.uuid4())
    artifact_name_list = [artifact_name]
    dist.broadcast_object_list(artifact_name_list)
    artifact_name = artifact_name_list[0]

    if dist.get_global_rank() == 0:
        # Create a dummy artifact
        dummy_artifact_path = tmp_path / 'artifact.txt'
        with open(dummy_artifact_path, 'w+') as f:
            f.write('hello!')

        # Log an artifact if rank zero
        logger.file_artifact(
            log_level=LogLevel.FIT,
            file_path=dummy_artifact_path,
            artifact_name=artifact_name,
        )

    # Wait for rank 0 queue the file upload
    dist.barrier()

    # Attempt to retrieve the artifact on all ranks
    downloaded_artifact_path = tmp_path / 'downloaded_artifact'

    @retry(FileNotFoundError, num_attempts=6)  # 6 attempts is ~2^(6-1) seconds max wait
    def _validate_artifact():
        wandb_logger.get_file_artifact(artifact_name, str(downloaded_artifact_path))
        with open(downloaded_artifact_path, 'r') as f:
            assert f.read() == 'hello!'

    _validate_artifact()
