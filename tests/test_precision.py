# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import torch.distributed
from packaging import version

import composer
from composer.core import Precision
from composer.datasets.hparams import SyntheticHparamsMixin
from composer.trainer import TrainerHparams
from composer.trainer.devices import GPUDeviceHparams


def run_and_measure_memory(precision: Precision) -> int:
    hparams_f = os.path.join(os.path.dirname(composer.__file__), "yamls", "models", "resnet56_cifar10_synthetic.yaml")
    hparams = TrainerHparams.create(f=hparams_f, cli_args=False)
    hparams.train_subset_num_batches = 1
    hparams.eval_interval = "0ep"
    assert isinstance(hparams, TrainerHparams)
    assert isinstance(hparams.device, GPUDeviceHparams)
    hparams.precision = precision
    hparams.dataloader.num_workers = 0
    hparams.dataloader.persistent_workers = False
    hparams.max_duration = "1ep"
    assert isinstance(hparams.train_dataset, SyntheticHparamsMixin)
    hparams.train_dataset.use_synthetic = True
    assert isinstance(hparams.val_dataset, SyntheticHparamsMixin)
    hparams.val_dataset.use_synthetic = True
    hparams.loggers = []
    trainer = hparams.initialize_object()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    trainer.fit()
    return torch.cuda.max_memory_allocated()


@pytest.mark.gpu
@pytest.mark.timeout(5)
@pytest.mark.parametrize("precision", [Precision.AMP, Precision.BF16])
def test_precision_memory(precision: Precision):
    if version.parse(torch.__version__) < version.parse("1.10"):
        pytest.skip("Test required torch >= 1.10")
    memory_full = run_and_measure_memory(Precision.FP32)
    memory_precision = run_and_measure_memory(precision)
    assert memory_precision < 0.7 * memory_full
