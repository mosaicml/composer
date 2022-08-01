# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict

import pytest
import yaml

import composer
from composer.core.precision import Precision
from composer.trainer.trainer_hparams import TrainerHparams
from tests.common import configure_dataset_hparams_for_synthetic, configure_model_hparams_for_synthetic


def walk_model_yamls():
    yamls = []
    for root, dirs, files in os.walk(os.path.join(os.path.dirname(composer.__file__), 'yamls')):
        del dirs  # unused
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('.yaml'):
                yamls.append(filepath)
    assert len(yamls) > 0, 'there should be at least one yaml!'
    return yamls


@pytest.mark.parametrize('hparams_file', walk_model_yamls())
class TestHparamsCreate:

    def _ensure_device_cpu(self, yaml_dict: Dict[str, Any]):
        yaml_dict['device'] = {'cpu': {}}
        yaml_dict['precision'] = 'fp32'
        return yaml_dict

    def test_hparams_create(self, hparams_file: str):
        if 'timm' in hparams_file:
            pytest.importorskip('timm')
        if 'vit' in hparams_file:
            pytest.importorskip('vit_pytorch')
        if hparams_file in ['unet.yaml']:
            pytest.importorskip('monai')
        if 'deeplabv3' in hparams_file:
            pytest.importorskip('mmseg')

        with open(hparams_file, 'r') as f:
            yaml_dict = yaml.safe_load(f)
            print(yaml_dict)

        yaml_dict = self._ensure_device_cpu(yaml_dict)

        hparams = TrainerHparams.create(data=yaml_dict, cli_args=False)
        assert isinstance(hparams, TrainerHparams)

    @pytest.mark.filterwarnings(
        r'ignore:Metric `SpearmanCorrcoef` will save all targets and predictions in the buffer:UserWarning:torchmetrics'
    )
    def test_trainer_initialize(self, hparams_file: str):
        if 'timm' in hparams_file:
            pytest.importorskip('timm')
        if 'vit' in hparams_file:
            pytest.importorskip('vit_pytorch')
        if 'glue/mnli.yaml' in hparams_file:
            pytest.xfail(
                'The max duration for MNLI, combined with the warmup period, results in a warmup duration of 0.')
        if hparams_file in ['unet.yaml']:
            pytest.importorskip('monai')

        nlp_hparam_keys = ['glue', 'gpt', 'bert']
        # skip tests that require the NLP stack
        if any([i in hparams_file for i in nlp_hparam_keys]):
            pytest.importorskip('transformers')
            pytest.importorskip('datasets')
            pytest.importorskip('tokenizers')

        if 'deeplabv3' in hparams_file:
            pytest.importorskip('mmseg')

        with open(hparams_file, 'r') as f:
            yaml_dict = yaml.safe_load(f)

        yaml_dict = self._ensure_device_cpu(yaml_dict)

        hparams = TrainerHparams.create(data=yaml_dict, cli_args=False)
        hparams.dataloader.num_workers = 0
        hparams.dataloader.persistent_workers = False
        hparams.dataloader.pin_memory = False
        hparams.dataloader.prefetch_factor = 2
        hparams.precision = Precision.FP32

        if hparams.train_dataset is not None:
            configure_dataset_hparams_for_synthetic(hparams.train_dataset, model_hparams=hparams.model)
        configure_model_hparams_for_synthetic(hparams.model)
        if hparams.val_dataset is not None:
            configure_dataset_hparams_for_synthetic(hparams.val_dataset)
        if hparams.evaluators is not None:
            for evaluator in hparams.evaluators:
                configure_dataset_hparams_for_synthetic(evaluator.eval_dataset)
        hparams.load_path = None

        hparams.initialize_object()
