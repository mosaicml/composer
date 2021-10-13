# Copyright 2021 MosaicML. All Rights Reserved.

import os

import pytest
import torch
import torch.distributed

import composer
from composer.core.types import Precision
from composer.trainer import TrainerHparams
from composer.trainer.ddp import FileStoreHparams
from composer.trainer.devices import GPUDeviceHparams


def run_and_measure_memory(precision: Precision, file_store_path: str) -> int:
    hparams_f = os.path.join(os.path.dirname(composer.__file__), "yamls", "models", "resnet56_cifar10",
                             "hparams_synthetic.yaml")
    hparams = TrainerHparams.create(f=hparams_f)
    assert isinstance(hparams, TrainerHparams)
    assert isinstance(hparams.device, GPUDeviceHparams)
    hparams.device.n_gpus = 1
    hparams.precision = precision
    hparams.ddp.store = FileStoreHparams(file_store_path)
    hparams.ddp.fork_rank_0 = False
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
@pytest.mark.run_long
@pytest.mark.n_gpus(1)
def test_fp16_mixed(ddp_tmpdir: str):
    memory_full = run_and_measure_memory(Precision.FP32, os.path.join(ddp_tmpdir, "store_full"))
    memory_amp = run_and_measure_memory(Precision.AMP, os.path.join(ddp_tmpdir, "store_amp"))
    assert memory_amp < 0.7 * memory_full
