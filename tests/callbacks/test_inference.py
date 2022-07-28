# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""
Test inference APIs.
"""
import os
import tempfile
from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader

from composer.callbacks import ExportForInferenceCallback
from composer.models import composer_resnet
from composer.trainer import Trainer
from tests.common.datasets import RandomImageDataset


@pytest.mark.parametrize(
    'model_cls, sample_input',
    [
        (partial(composer_resnet, 'resnet18'), (torch.rand(1, 3, 224, 224), torch.randint(10, (1,)))),
    ],
)
def test_inference_callback_torchscript(model_cls, sample_input):
    save_format = 'torchscript'
    model = model_cls()

    with tempfile.TemporaryDirectory() as tempdir:
        save_path = os.path.join(tempdir, f'model.pt')
        export_for_inference = ExportForInferenceCallback(save_format=save_format, save_path=str(save_path))

        # Construct the trainer and train
        trainer = Trainer(
            model=model,
            callbacks=export_for_inference,
            train_dataloader=DataLoader(RandomImageDataset(shape=(3, 224, 224))),
            max_duration='1ba',
        )
        trainer.fit()

        model.eval()
        orig_out = model(sample_input)

        loaded_model = torch.jit.load(save_path)
        loaded_model.eval()
        loaded_model_out = loaded_model(sample_input)

        torch.testing.assert_close(orig_out, loaded_model_out)


@pytest.mark.parametrize(
    'model_cls, sample_input',
    [
        (partial(composer_resnet, 'resnet18'), (torch.rand(1, 3, 224, 224), torch.randint(10, (1,)))),
    ],
)
def test_inference_callback_onnx(model_cls, sample_input):
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    import onnx
    import onnxruntime as ort

    save_format = 'onnx'
    model = model_cls()

    with tempfile.TemporaryDirectory() as tempdir:
        save_path = os.path.join(tempdir, f'model.onnx')
        export_for_inference = ExportForInferenceCallback(save_format=save_format, save_path=str(save_path))

        # Construct the trainer and train
        trainer = Trainer(
            model=model,
            callbacks=export_for_inference,
            train_dataloader=DataLoader(RandomImageDataset(shape=(3, 224, 224))),
            max_duration='1ba',
        )
        trainer.fit()

        model.eval()
        orig_out = model(sample_input)

        loaded_model = onnx.load(save_path)
        onnx.checker.check_model(loaded_model)

        ort_session = ort.InferenceSession(save_path)
        loaded_model_out = ort_session.run(
            None,
            {'input': sample_input[0].numpy()},
        )

        torch.testing.assert_close(
            orig_out.detach().numpy(),
            loaded_model_out[0],
            rtol=1e-4,  # lower tolerance for ONNX
            atol=1e-3,  # lower tolerance for ONNX
        )
