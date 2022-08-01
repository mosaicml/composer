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

from composer.models import composer_resnet
from composer.utils import export_for_inference


@pytest.mark.parametrize(
    'model_cls, sample_input',
    [
        (partial(composer_resnet, 'resnet18'), (torch.rand(4, 3, 224, 224), torch.randint(10, (4,)))),
    ],
)
def test_export_for_inference_torchscript(model_cls, sample_input):
    model = model_cls()
    model.eval()

    orig_out = model(sample_input)

    save_format = 'torchscript'
    with tempfile.TemporaryDirectory() as tempdir:
        save_path = os.path.join(tempdir, f'model.pt')
        export_for_inference(model=model, save_format=save_format, save_path=save_path)
        loaded_model = torch.jit.load(save_path)
        loaded_model.eval()
        loaded_model_out = loaded_model(sample_input)

        assert torch.allclose(
            orig_out,
            loaded_model_out), f'mismatch in the original and exported for inference model outputs with {save_format}'


@pytest.mark.parametrize(
    'model_cls, sample_input',
    [
        (partial(composer_resnet, 'resnet18'), (torch.rand(4, 3, 224, 224), torch.randint(10, (4,)))),
    ],
)
def test_export_for_inference_onnx(model_cls, sample_input):
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    import onnx  # type: ignore
    import onnxruntime as ort  # type: ignore

    model = model_cls()
    model.eval()

    orig_out = model(sample_input)

    save_format = 'onnx'
    with tempfile.TemporaryDirectory() as tempdir:
        save_path = os.path.join(tempdir, f'model.{save_format}')
        export_for_inference(model=model, save_format=save_format, save_path=save_path, sample_input=(sample_input,))

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
            msg=f'mismatch in the original and exported for inference model outputs with {save_format}')
