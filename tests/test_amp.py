# Copyright 2021 MosaicML. All Rights Reserved.

import os

import pytest
import torch
import torch.distributed

import composer
from composer.core.types import Precision
from composer.trainer import TrainerHparams
from composer.trainer.devices import GPUDeviceHparams


def run_and_measure_memory(precision: Precision) -> int:
    hparams_f = os.path.join(os.path.dirname(composer.__file__), "yamls", "models", "resnet56_cifar10",
                             "hparams_synthetic.yaml")
    hparams = TrainerHparams.create(f=hparams_f, cli_args=False)
    assert isinstance(hparams, TrainerHparams)
    assert isinstance(hparams.device, GPUDeviceHparams)
    hparams.precision = precision
    hparams.dataloader.num_workers = 0
    hparams.dataloader.persistent_workers = False
    hparams.max_epochs = 2
    hparams.loggers = []
    trainer = hparams.initialize_object()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    trainer.fit()
    return torch.cuda.max_memory_allocated()


@pytest.mark.timeout(60)
@pytest.mark.gpu
def test_fp16_mixed(ddp_tmpdir: str):
    memory_full = run_and_measure_memory(Precision.FP32)
    memory_amp = run_and_measure_memory(Precision.AMP)
    assert memory_amp < 0.7 * memory_full
