# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import torch.distributed

import composer
from composer.core import Precision
from composer.datasets.synthetic_hparams import SyntheticHparamsMixin
from composer.trainer.trainer_hparams import TrainerHparams


def setup_hparams(precision: Precision) -> TrainerHparams:
    hparams_f = os.path.join(os.path.dirname(composer.__file__), 'yamls', 'models', 'resnet56_cifar10_synthetic.yaml')
    hparams = TrainerHparams.create(f=hparams_f, cli_args=False)
    hparams.train_subset_num_batches = 1
    hparams.eval_interval = '1ep'
    assert isinstance(hparams, TrainerHparams)
    hparams.precision = precision
    hparams.dataloader.num_workers = 0
    hparams.dataloader.persistent_workers = False
    hparams.max_duration = '1ep'
    assert isinstance(hparams.train_dataset, SyntheticHparamsMixin)
    hparams.train_dataset.use_synthetic = True
    assert isinstance(hparams.val_dataset, SyntheticHparamsMixin)
    hparams.val_dataset.use_synthetic = True
    hparams.loggers = []
    return hparams


def train_run_and_measure_memory(precision: Precision) -> int:
    hparams = setup_hparams(precision)
    trainer = hparams.initialize_object()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    trainer.fit()
    return torch.cuda.max_memory_allocated()


@pytest.mark.gpu
@pytest.mark.parametrize('precision', [Precision.AMP, Precision.BF16])
def test_train_precision_memory(precision: Precision):
    memory_full = train_run_and_measure_memory(Precision.FP32)
    memory_precision = train_run_and_measure_memory(precision)
    assert memory_precision < 0.7 * memory_full


def eval_run_and_measure_memory(precision: Precision) -> int:
    hparams = setup_hparams(precision)
    trainer = hparams.initialize_object()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    trainer.eval(
        dataloader=trainer.state.evaluators[0].dataloader,
        dataloader_label='eval',
        metrics=trainer.state.evaluators[0].metrics,
    )
    return torch.cuda.max_memory_allocated()


@pytest.mark.gpu
@pytest.mark.parametrize('precision', [Precision.AMP, Precision.BF16])
def test_eval_precision_memory(precision: Precision):
    memory_full = eval_run_and_measure_memory(Precision.FP32)
    memory_precision = eval_run_and_measure_memory(precision)
    assert memory_precision < 0.9 * memory_full


def predict_run_and_measure_memory(precision: Precision) -> int:
    hparams = setup_hparams(precision)
    trainer = hparams.initialize_object()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    trainer.predict(dataloader=trainer.state.evaluators[0].dataloader,)
    return torch.cuda.max_memory_allocated()


@pytest.mark.gpu
@pytest.mark.parametrize('precision', [Precision.AMP, Precision.BF16])
def test_predict_precision_memory(precision: Precision):
    memory_full = predict_run_and_measure_memory(Precision.FP32)
    memory_precision = predict_run_and_measure_memory(precision)
    assert memory_precision < 0.9 * memory_full
