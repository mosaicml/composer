# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import glob
import os
from typing import List, Optional, Type, Union

import pytest
import yahp as hp

import composer
from composer import Algorithm, Trainer
from composer.algorithms.algorithm_hparams_registry import algorithm_registry
from composer.core.precision import Precision
from composer.trainer.devices.device_cpu import DeviceCPU
from composer.trainer.trainer_hparams import TrainerHparams
from tests.common import configure_dataset_hparams_for_synthetic, configure_model_hparams_for_synthetic

modeldir_path = os.path.join(os.path.dirname(composer.__file__), 'yamls', 'models')
model_names = glob.glob(os.path.join(modeldir_path, '*.yaml'))
model_names = [os.path.basename(os.path.splitext(mn)[0]) for mn in model_names]


def load(algorithm_cls: Union[Type[Algorithm], Type[hp.Hparams]], alg_params: Optional[str]) -> Algorithm:
    inverted_registry = {v: k for (k, v) in algorithm_registry.items()}
    alg_name = inverted_registry[algorithm_cls]
    alg_folder = os.path.join(os.path.dirname(composer.__file__), 'yamls', 'algorithms')
    if alg_params is None:
        hparams_file = os.path.join(alg_folder, f'{alg_name}.yaml')
    else:
        hparams_file = os.path.join(alg_folder, alg_name, f'{alg_params}.yaml')
    alg = hp.create(algorithm_cls, f=hparams_file, cli_args=False)
    assert isinstance(alg, Algorithm)
    return alg


def load_multiple(cls, *algorithms: str) -> List[Algorithm]:
    algs = []
    for alg in algorithms:
        alg_parts = alg.split('/')
        alg_name = alg_parts[0]
        if len(alg_parts) > 1:
            alg_params = '/'.join(alg_parts[1:])
        else:
            alg_params = None
        try:
            alg = algorithm_registry[alg_name]
        except KeyError as e:
            raise ValueError(f'Algorithm {e.args[0]} not found') from e
        algs.append(load(alg, alg_params))
    return algs


def get_model_algs(model_name: str) -> List[str]:
    algs = list(algorithm_registry.keys())
    algs.remove('no_op_model')

    is_image_model = any(x in model_name for x in ('resnet', 'mnist', 'efficientnet', 'timm', 'vit', 'deeplabv3'))
    is_language_model = any(x in model_name for x in ('gpt2', 'bert'))

    if 'vit_small_patch16' in model_name:
        algs.remove('blurpool')

    if 'classify_mnist' in model_name:
        # Mnist has no strided conv2d or pool2d
        # see https://github.com/pytorch/examples/blob/41b035f2f8faede544174cfd82960b7b407723eb/mnist/main.py#L14
        algs.remove('blurpool')

    if is_image_model:
        algs.remove('alibi')
        algs.remove('seq_length_warmup')
        algs.remove('swa')
    if 'alibi' in algs:
        pytest.importorskip('transformers')
        pytest.importorskip('datasets')
        pytest.importorskip('tokenizers')
    if is_language_model:
        algs.remove('blurpool')
        algs.remove('channels_last')
        algs.remove('cutmix')
        algs.remove('cutout')
        algs.remove('factorize')
        algs.remove('ghost_batchnorm')
        algs.remove('label_smoothing')
        algs.remove('layer_freezing')
        algs.remove('squeeze_excite')
        algs.remove('swa')
        algs.remove('mixup')
        algs.remove('stochastic_depth')
        algs.remove('colout')
        algs.remove('progressive_resizing')
        algs.remove('randaugment')
        algs.remove('augmix')
        algs.remove('sam')
        algs.remove('selective_backprop')
    if model_name in ('unet'):
        algs.remove('stochastic_depth')
        algs.remove('mixup')
        algs.remove('cutmix')
    return algs


@pytest.mark.parametrize('model_name', model_names)
@pytest.mark.timeout(15)
@pytest.mark.filterwarnings(
    r'ignore:Metric `SpearmanCorrcoef` will save all targets and predictions in the buffer:UserWarning:torchmetrics')
def test_load(model_name: str):
    if 'timm' in model_name:
        pytest.importorskip('timm')
    if 'vit' in model_name:
        pytest.importorskip('vit_pytorch')
    if model_name in ['unet']:
        pytest.importorskip('monai')
    if model_name in ['deeplabv3_ade20k_unoptimized', 'deeplabv3_ade20k_optimized']:
        pytest.importorskip('mmcv')
        pytest.skip(f'Model {model_name} requires GPU')

    trainer_hparams = TrainerHparams.load(model_name)
    trainer_hparams.precision = Precision.FP32
    trainer_hparams.algorithms = load_multiple(*get_model_algs(model_name))

    assert trainer_hparams.train_dataset is not None
    configure_dataset_hparams_for_synthetic(trainer_hparams.train_dataset, model_hparams=trainer_hparams.model)
    trainer_hparams.train_subset_num_batches = 1

    # Only one of val_dataset or evaluators should be set
    assert trainer_hparams.val_dataset is not None or trainer_hparams.evaluators is not None
    assert trainer_hparams.val_dataset is None or trainer_hparams.evaluators is None
    if trainer_hparams.evaluators is not None:
        for evaluator in trainer_hparams.evaluators:
            configure_dataset_hparams_for_synthetic(evaluator.eval_dataset)
    if trainer_hparams.val_dataset is not None:
        configure_dataset_hparams_for_synthetic(trainer_hparams.val_dataset)

    trainer_hparams.eval_subset_num_batches = 1

    trainer_hparams.dataloader.num_workers = 0
    trainer_hparams.dataloader.pin_memory = False
    trainer_hparams.dataloader.prefetch_factor = 2
    trainer_hparams.dataloader.persistent_workers = False

    configure_model_hparams_for_synthetic(trainer_hparams.model)

    trainer_hparams.device = DeviceCPU()
    my_trainer = trainer_hparams.initialize_object()

    assert isinstance(my_trainer, Trainer)
