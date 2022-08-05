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
import torch.nn as nn
from torch.utils.data import DataLoader

from composer.core import State
from composer.models import composer_resnet
from composer.trainer.ddp import prepare_ddp_module
from composer.utils import dist, export_for_inference
from tests.common.datasets import RandomImageDataset


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
    import onnx
    import onnx.checker
    import onnxruntime as ort

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


@pytest.mark.parametrize(
    'model_cls, sample_input',
    [
        (partial(composer_resnet, 'resnet18'), (torch.rand(1, 3, 224, 224), torch.randint(10, (1,)))),
    ],
)
@pytest.mark.world_size(2)
def test_export_for_inference_onnx_ddp(model_cls, sample_input):
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    import onnx
    import onnx.checker
    import onnxruntime as ort

    model = model_cls()

    optimizer = torch.optim.SGD(model.parameters(), 0.1)

    state = State(
        model=model,
        rank_zero_seed=0,
        run_name='run_name',
        optimizers=optimizer,
        grad_accum=2,
        max_duration='1ep',
        dataloader=DataLoader(RandomImageDataset(shape=(3, 224, 224))),
        dataloader_label='train',
        precision='fp32',
    )

    state.model = prepare_ddp_module(state.model, find_unused_parameters=True)
    state.model.eval()
    orig_out = state.model(sample_input)

    save_format = 'onnx'

    # Only one rank needs to save/load model
    if dist.get_local_rank() == 0:
        with tempfile.TemporaryDirectory() as tempdir:
            save_path = os.path.join(str(tempdir), f'model.{save_format}')
            assert isinstance(state.model.module, nn.Module)
            export_for_inference(model=state.model.module,
                                 save_format=save_format,
                                 save_path=save_path,
                                 sample_input=(sample_input,))

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


@pytest.mark.parametrize(
    'model_cls, sample_input',
    [
        (partial(composer_resnet, 'resnet18'), (torch.rand(1, 3, 224, 224), torch.randint(10, (1,)))),
    ],
)
@pytest.mark.world_size(2)
def test_export_for_inference_torchscript_ddp(model_cls, sample_input):
    model = model_cls()

    optimizer = torch.optim.SGD(model.parameters(), 0.1)

    state = State(
        model=model,
        rank_zero_seed=0,
        run_name='run_name',
        optimizers=optimizer,
        grad_accum=2,
        max_duration='1ep',
        dataloader=DataLoader(RandomImageDataset(shape=(3, 224, 224))),
        dataloader_label='train',
        precision='fp32',
    )

    state.model = prepare_ddp_module(state.model, find_unused_parameters=True)
    state.model.eval()
    orig_out = state.model(sample_input)

    save_format = 'torchscript'

    # Only one rank needs to save/load model
    if dist.get_local_rank() == 0:
        with tempfile.TemporaryDirectory() as tempdir:
            save_path = os.path.join(str(tempdir), f'model.pt')
            assert isinstance(state.model.module, nn.Module)
            export_for_inference(model=state.model.module, save_format=save_format, save_path=save_path)

            loaded_model = torch.jit.load(save_path)
            loaded_model.eval()
            loaded_model_out = loaded_model(sample_input)

            torch.testing.assert_close(orig_out, loaded_model_out)
