# Copyright 2021 MosaicML. All Rights Reserved.

import itertools

import pytest
import torch

from composer.algorithms.stratify_batches import StratifiedBatchSampler, StratifyBatches, StratifyBatchesHparams
from composer.core import Event, Logger, State, surgery
from composer.core.algorithm import Algorithm
from composer.core.types import Tensors
from composer.datasets.dataloader import DataloaderHparams
from composer.trainer import Trainer
from composer.trainer.trainer_hparams import TrainerHparams
from tests.utils.trainer_fit import train_model


@pytest.fixture(params=['match', 'balance'])
# @pytest.fixture(params=['balance'])
def algo_hparams_instance(request):
    return StratifyBatchesHparams(stratify_how=request.param, targets_attr='input_target')


@pytest.mark.timeout(5)
def test_correct_sampler_created(mosaic_trainer_hparams: TrainerHparams, algo_hparams_instance: StratifyBatchesHparams):
    mosaic_trainer_hparams.algorithms = [algo_hparams_instance]
    mosaic_trainer_hparams.train_dataset.shuffle = True  # shuffle=False not implemented
    assert mosaic_trainer_hparams.train_dataset.is_train
    assert not mosaic_trainer_hparams.val_dataset.is_train
    trainer = Trainer.create_from_hparams(mosaic_trainer_hparams)
    assert isinstance(trainer.state.train_dataloader.batch_sampler, StratifiedBatchSampler)
    assert not isinstance(trainer.state.eval_dataloader.batch_sampler, StratifiedBatchSampler)


# @pytest.mark.timeout(5)
# def test_algo_trains(mosaic_trainer_hparams: TrainerHparams, algo_hparams_instance: StratifyBatchesHparams):
#     mosaic_trainer_hparams.algorithms = [algo_hparams_instance]
#     train_model(mosaic_trainer_hparams, run_loss_check=True)
