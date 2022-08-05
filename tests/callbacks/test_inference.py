# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""
Test inference APIs.
"""
import os
import tempfile
from functools import partial
from unittest.mock import patch

import pytest
from torch.utils.data import DataLoader

from composer.callbacks import ExportForInferenceCallback, export_for_inference
from composer.models import composer_resnet
from composer.trainer import Trainer
from tests.common.datasets import RandomImageDataset


@pytest.mark.parametrize(
    'model_cls',
    [partial(composer_resnet, 'resnet18')],
)
def test_inference_callback_torchscript(model_cls):
    with patch('composer.callbacks.export_for_inference.export_with_logger'):
        save_format = 'torchscript'
        model = model_cls()

        with tempfile.TemporaryDirectory() as tempdir:
            save_path = os.path.join(tempdir, f'model.pt')
            exp_for_inf_callback = ExportForInferenceCallback(save_format=save_format, save_path=str(save_path))

            # Construct the trainer and train
            trainer = Trainer(
                model=model,
                callbacks=exp_for_inf_callback,
                train_dataloader=DataLoader(RandomImageDataset(shape=(3, 224, 224))),
                max_duration='1ba',
            )
            trainer.fit()

            # Assert export_for_inference utility called with expected inputs
            export_for_inference.export_with_logger.assert_called_once_with(
                model=model,
                save_format=save_format,
                save_path=save_path,
                logger=trainer.logger,
                save_object_store=None,
                sample_input=(exp_for_inf_callback.sample_input,),
                transforms=None)


@pytest.mark.parametrize(
    'model_cls',
    [partial(composer_resnet, 'resnet18')],
)
def test_inference_callback_onnx(model_cls):
    with patch('composer.callbacks.export_for_inference.export_with_logger'):
        save_format = 'onnx'
        model = model_cls()

        with tempfile.TemporaryDirectory() as tempdir:
            save_path = os.path.join(tempdir, f'model.onnx')
            exp_for_inf_callback = ExportForInferenceCallback(save_format=save_format, save_path=str(save_path))

            # Construct the trainer and train
            trainer = Trainer(model=model,
                              callbacks=exp_for_inf_callback,
                              train_dataloader=DataLoader(RandomImageDataset(shape=(3, 224, 224))),
                              max_duration='1ba')
            trainer.fit()

            # Assert export_for_inference utility called with expected inputs
            export_for_inference.export_with_logger.assert_called_once_with(
                model=model,
                save_format=save_format,
                save_path=save_path,
                logger=trainer.logger,
                save_object_store=None,
                sample_input=(exp_for_inf_callback.sample_input,),
                transforms=None)
