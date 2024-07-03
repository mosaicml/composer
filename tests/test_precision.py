# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional

import pytest
import torch
import torch.distributed
from torch.utils.data import DataLoader

from composer import Trainer
from composer.core import Precision, get_precision_context
from tests.common import RandomImageDataset, composer_resnet

try:
    import transformer_engine.pytorch as te
    te_installed = True
except ImportError:
    te_installed = False


def get_trainer(precision: Precision, precision_config: Optional[dict[str, Any]] = None) -> Trainer:

    return Trainer(
        model=composer_resnet('resnet18'),
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
        precision_config=precision_config,
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
@pytest.mark.filterwarnings(r'ignore:.*Plan failed with a cudnnException.*:UserWarning')  # Torch 2.3 regression
def test_train_precision_memory(precision: Precision):
    memory_fp32 = fit_and_measure_memory(Precision.FP32)
    memory_half = fit_and_measure_memory(precision)
    assert memory_half < 0.87 * memory_fp32


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


@pytest.mark.gpu
def test_amp_fp8_config():
    if te_installed and torch.cuda.get_device_capability()[0] >= 9:
        from transformer_engine.common.recipe import Format
        precision_config = {
            'fp8_format': Format.HYBRID,
            'amax_history_len': 16,
            'amax_compute_algo': 'max',
        }
        trainer = get_trainer(Precision.AMP_FP8, precision_config=precision_config)
        with get_precision_context(trainer.state.precision, trainer.state.precision_config):
            fp8_recipe = te.fp8.get_fp8_recipe()
            for k, v in precision_config.items():
                assert getattr(fp8_recipe, k) == v
