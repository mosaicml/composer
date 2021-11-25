# Copyright 2021 MosaicML. All Rights Reserved.

import glob
import os
from typing import List

import pytest

import composer
import composer.algorithms as algorithms
import composer.trainer as trainer
from composer.algorithms.scale_schedule.scale_schedule import ScaleScheduleHparams
from composer.core.precision import Precision
from composer.trainer.devices.device_hparams import CPUDeviceHparams

modeldir_path = os.path.join(os.path.dirname(composer.__file__), 'yamls', 'models')
model_names = glob.glob(os.path.join(modeldir_path, '*.yaml'))
model_names = [os.path.basename(os.path.splitext(mn)[0]) for mn in model_names]


def get_model_algs(model_name: str) -> List[str]:
    algs = algorithms.list_algorithms()
    is_image_model = any(x in model_name for x in ("resnet", "mnist", "efficientnet"))
    if is_image_model:
        algs.remove("alibi")
    if model_name in ("unet", "gpt2_52m", "gpt2_83m", 'gpt2_125m'):
        algs.remove("mixup")
        algs.remove("cutmix")
    return algs


@pytest.mark.parametrize('model_name', model_names)
@pytest.mark.timeout(15)
def test_load(model_name: str):
    trainer_hparams = trainer.load(model_name)
    trainer_hparams.precision = Precision.FP32
    trainer_hparams.algorithms = algorithms.load_multiple(*get_model_algs(model_name))
    try:
        trainer_hparams.train_dataset.set_num_total_batches(1)
    except NotImplementedError:
        pytest.xfail(f"Model {model_name} uses a train dataset that doesn't support num_total_batches")
        raise

    try:
        train_synthetic = trainer_hparams.train_dataset.get_synthetic()
    except NotImplementedError:
        pytest.xfail(f"Model {model_name} uses a train dataset that doesn't support synthetic data.")
        raise
    if train_synthetic is None:
        trainer_hparams.train_dataset.set_synthetic(trainer_hparams.train_dataset.get_synthetic_hparams_cls()())

    try:
        trainer_hparams.val_dataset.set_num_total_batches(1)
    except NotImplementedError:
        pytest.xfail(f"Model {model_name} uses a val dataset that doesn't support num_total_batches")
        raise

    try:
        val_synthetic = trainer_hparams.val_dataset.get_synthetic()
    except NotImplementedError:
        pytest.xfail(f"Model {model_name} uses a val dataset that doesn't support synthetic data.")
        raise

    if val_synthetic is None:
        trainer_hparams.val_dataset.set_synthetic(trainer_hparams.val_dataset.get_synthetic_hparams_cls()())
    trainer_hparams.device = CPUDeviceHparams()
    my_trainer = trainer_hparams.initialize_object()

    assert isinstance(my_trainer, trainer.Trainer)


@pytest.mark.parametrize("ssr", ["0.25", "0.33", "0.50", "0.67", "0.75", "1.00", "1.25"])
def test_scale_schedule_load(ssr: str):
    trainer_hparams = trainer.load("classify_mnist")
    trainer_hparams.precision = Precision.FP32
    algs = [f"scale_schedule/{ssr}"]
    trainer_hparams.algorithms = algorithms.load_multiple(*algs)
    trainer_hparams.train_dataset.set_num_total_batches(1)
    if trainer_hparams.train_dataset.get_synthetic() is None:
        trainer_hparams.train_dataset.set_synthetic(trainer_hparams.train_dataset.get_synthetic_hparams_cls()())
    trainer_hparams.val_dataset.set_num_total_batches(1)
    if trainer_hparams.val_dataset.get_synthetic() is None:
        trainer_hparams.val_dataset.set_synthetic(trainer_hparams.val_dataset.get_synthetic_hparams_cls()())
    trainer_hparams.device = CPUDeviceHparams()
    assert len(trainer_hparams.algorithms) == 1
    alg = trainer_hparams.algorithms[0]
    assert isinstance(alg, ScaleScheduleHparams)
    assert alg.ratio == float(ssr)
    my_trainer = trainer_hparams.initialize_object()
    assert isinstance(my_trainer, trainer.Trainer)
