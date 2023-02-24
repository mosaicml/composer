# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.distributed
from torch.utils.data import DataLoader

from composer import Trainer
from composer.core import Precision
from composer.models import composer_resnet_cifar
from tests.common import RandomImageDataset

try:
    import transformer_engine.pytorch as te
    del te
    te_installed = True
except ImportError:
    te_installed = False


def get_trainer(precision: Precision) -> Trainer:

    return Trainer(
        model=composer_resnet_cifar('resnet_9'),
        train_dataloader=DataLoader(
            dataset=RandomImageDataset(size=1024),
            batch_size=512,
            persistent_workers=False,
            num_workers=0,
        ),
        eval_dataloader=DataLoader(
            dataset=RandomImageDataset(size=1024),
            batch_size=512,
            persistent_workers=False,
            num_workers=0,
        ),
        precision=precision,
        max_duration='1ep',
        eval_interval='1ep',
        train_subset_num_batches=1,
    )


def fit_and_measure_memory(precision) -> int:
    trainer = get_trainer(precision)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    trainer.fit()

    return torch.cuda.max_memory_allocated()


def eval_and_measure_memory(precision) -> int:
    trainer = get_trainer(precision)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    trainer.eval()

    return torch.cuda.max_memory_allocated()


def predict_and_measure_memory(precision) -> int:
    trainer = get_trainer(precision)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    trainer.predict(dataloader=trainer.state.evaluators[0].dataloader)

    return torch.cuda.max_memory_allocated()


@pytest.mark.gpu
@pytest.mark.parametrize('precision', [Precision.AMP_FP16, Precision.AMP_BF16])
def test_train_precision_memory(precision: Precision):
    memory_fp32 = fit_and_measure_memory(Precision.FP32)
    memory_half = fit_and_measure_memory(precision)
    assert memory_half < 0.7 * memory_fp32


@pytest.mark.gpu
@pytest.mark.parametrize('precision', [Precision.AMP_FP16, Precision.AMP_BF16])
def test_eval_precision_memory(precision: Precision):
    memory_fp32 = eval_and_measure_memory(Precision.FP32)
    memory_half = eval_and_measure_memory(precision)
    assert memory_half < 0.95 * memory_fp32


@pytest.mark.gpu
@pytest.mark.parametrize('precision', [Precision.AMP_FP16, Precision.AMP_BF16])
def test_predict_precision_memory(precision: Precision):
    memory_fp32 = predict_and_measure_memory(Precision.FP32)
    memory_half = predict_and_measure_memory(precision)
    assert memory_half < 0.95 * memory_fp32


@pytest.mark.gpu
def test_amp_fp8_path():
    trainer = get_trainer(Precision.AMP_FP8)
    if te_installed:
        if torch.cuda.get_device_capability()[0] < 9:
            with pytest.raises(RuntimeError, match='AMP_FP8 precision is used but current device does not support it'):
                trainer.fit()
        else:
            trainer.fit()
    else:
        with pytest.raises(ImportError, match='AMP_FP8 precision is used but TransformerEngine is not installed'):
            trainer.fit()
