# Copyright 2021 MosaicML. All Rights Reserved.

import glob
import os

import pytest

import composer
import composer.algorithms as algorithms
import composer.trainer as trainer
from composer.algorithms.scale_schedule.scale_schedule import ScaleScheduleHparams
from composer.core.precision import Precision
from composer.datasets import SyntheticDatasetHparams
from composer.trainer.devices.device_hparams import CPUDeviceHparams

# skip loading models from huggingface, and resnet101 timeout
EXCLUDE_MODELS = ['gpt2_114m', 'gpt2_85m', 'gpt2_38m']

modeldir_path = os.path.join(os.path.dirname(composer.__file__), 'yamls', 'models')
model_names = glob.glob(os.path.join(modeldir_path, '*.yaml'))
model_names = [os.path.basename(os.path.splitext(mn)[0]) for mn in model_names]
model_names = [name for name in model_names if name not in EXCLUDE_MODELS]


@pytest.mark.parametrize('model_name', model_names)
def test_load(model_name: str, dummy_dataset_hparams: SyntheticDatasetHparams):
    if "gpt" in model_name:
        pytest.skip("GPT doesn't work on the no-op model class")
    if "unet" in model_name:
        pytest.skip("unet doesn't work on the no-op model class")

    dummy_dataset_hparams.sample_pool_size = 4096

    trainer_hparams = trainer.load(model_name)
    trainer_hparams.precision = Precision.FP32
    algs = algorithms.list_algorithms()
    trainer_hparams.algorithms = algorithms.load_multiple(*algs)
    trainer_hparams.train_dataset = dummy_dataset_hparams
    trainer_hparams.val_dataset = dummy_dataset_hparams
    trainer_hparams.device = CPUDeviceHparams(1)
    my_trainer = trainer_hparams.initialize_object()

    assert isinstance(my_trainer, trainer.Trainer)


@pytest.mark.parametrize("ssr", ["0.25", "0.33", "0.50", "0.67", "0.75", "1.00", "1.25"])
def test_scale_schedule_load(ssr: str, dummy_dataset_hparams: SyntheticDatasetHparams):
    dummy_dataset_hparams.sample_pool_size = 4096
    trainer_hparams = trainer.load("classify_mnist")
    trainer_hparams.precision = Precision.FP32
    algs = [f"scale_schedule/{ssr}"]
    trainer_hparams.algorithms = algorithms.load_multiple(*algs)
    trainer_hparams.train_dataset = dummy_dataset_hparams
    trainer_hparams.val_dataset = dummy_dataset_hparams
    trainer_hparams.device = CPUDeviceHparams(1)
    assert len(trainer_hparams.algorithms) == 1
    alg = trainer_hparams.algorithms[0]
    assert isinstance(alg, ScaleScheduleHparams)
    assert alg.ratio == float(ssr)
    my_trainer = trainer_hparams.initialize_object()
    assert isinstance(my_trainer, trainer.Trainer)
