# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""
Test inference APIs.
"""
import os
import tempfile
from functools import partial
from unittest.mock import ANY, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from composer.core import State
from composer.loggers import InMemoryLogger, Logger
from composer.loggers.logger_destination import LoggerDestination
from composer.models import composer_resnet
from composer.trainer.ddp import prepare_ddp_module
from composer.trainer.trainer import Trainer
from composer.utils import dist, export_with_logger, inference
from tests.common.datasets import RandomImageDataset


class MockFileArtifactLogger(LoggerDestination):
    """Mocks a generic file artifact logger interface."""

    def can_log_file_artifacts(self) -> bool:
        return True


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
        inference.export_for_inference(model=model, save_format=save_format, save_path=save_path)
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
        inference.export_for_inference(model=model,
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
            inference.export_for_inference(model=state.model.module,
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
            inference.export_for_inference(model=state.model.module, save_format=save_format, save_path=save_path)

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
def test_export_with_file_artifact_logger(model_cls, sample_input):
    with patch('composer.utils.inference.export_for_inference'):
        save_format = 'torchscript'
        model = model_cls()
        mock_obj_logger = MockFileArtifactLogger()
        with tempfile.TemporaryDirectory() as tempdir:
            save_path = os.path.join(tempdir, f'model.pt')

            # Construct the trainer and train
            trainer = Trainer(model=model,
                              train_dataloader=DataLoader(RandomImageDataset(shape=(3, 224, 224))),
                              max_duration='1ba')
            trainer.fit()

            mock_logger = Logger(state=trainer.state, destinations=[mock_obj_logger])

            export_with_logger(model=model,
                               save_format=save_format,
                               save_path=save_path,
                               sample_input=(sample_input,),
                               logger=mock_logger)

            # Assert export_for_inference utility called with expected inputs
            inference.export_for_inference.assert_called_once_with(model=model,
                                                                   save_format=save_format,
                                                                   save_path=ANY,
                                                                   sample_input=ANY,
                                                                   transforms=None)


@pytest.mark.parametrize(
    'model_cls, sample_input',
    [
        (partial(composer_resnet, 'resnet18'), (torch.rand(1, 3, 224, 224), torch.randint(10, (1,)))),
    ],
)
def test_export_with_other_logger(model_cls, sample_input):
    with patch('composer.utils.inference.export_for_inference'):
        save_format = 'torchscript'
        model = model_cls()
        non_file_artifact_logger = InMemoryLogger()
        with tempfile.TemporaryDirectory() as tempdir:
            save_path = os.path.join(tempdir, f'model.pt')

            # Construct the trainer and train
            trainer = Trainer(model=model,
                              train_dataloader=DataLoader(RandomImageDataset(shape=(3, 224, 224))),
                              max_duration='1ba')
            trainer.fit()

            mock_logger = Logger(state=trainer.state, destinations=[non_file_artifact_logger])

            export_with_logger(model=model,
                               save_format=save_format,
                               save_path=save_path,
                               sample_input=(sample_input,),
                               logger=mock_logger)

            # Assert export_for_inference utility called with expected inputs
            inference.export_for_inference.assert_called_once_with(model=model,
                                                                   save_format=save_format,
                                                                   save_path=save_path,
                                                                   save_object_store=None,
                                                                   sample_input=ANY,
                                                                   transforms=None)
