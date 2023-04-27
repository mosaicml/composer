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
from packaging import version
from torch.utils.data import DataLoader

from composer.core import State
from composer.devices import DeviceCPU, DeviceGPU
from composer.functional import apply_gated_linear_units
from composer.loggers import InMemoryLogger, Logger
from composer.loggers.logger_destination import LoggerDestination
from composer.models import composer_resnet
from composer.trainer.dist_strategy import prepare_ddp_module
from composer.trainer.trainer import Trainer
from composer.utils import dist, export_with_logger, inference
from composer.utils.device import get_device
from tests.common import SimpleTransformerClassifier, device
from tests.common.datasets import (RandomImageDataset, dummy_text_classification_dataloader, dummy_tiny_bert_lm_batch,
                                   dummy_transformer_classifier_batch)
from tests.common.models import configure_tiny_bert_hf_model


class MockFileUploader(LoggerDestination):
    """Mocks a generic file uploader interface."""

    def can_upload_files(self) -> bool:
        return True


@pytest.mark.parametrize(
    'model_cls, sample_input',
    [(partial(composer_resnet, 'resnet18'), (torch.rand(4, 3, 224, 224), torch.randint(10, (4,)))),
     pytest.param(SimpleTransformerClassifier, dummy_transformer_classifier_batch()),
     pytest.param(configure_tiny_bert_hf_model,
                  dummy_tiny_bert_lm_batch(),
                  marks=pytest.mark.xfail(reason='TinyBert HuggingFace model does not support torch.jit.script()'))],
)
def test_export_for_inference_torchscript(model_cls, sample_input):
    model = model_cls()

    model.eval()

    orig_out = model(sample_input)
    save_format = 'torchscript'
    with tempfile.TemporaryDirectory() as tempdir:
        save_path = os.path.join(tempdir, f'model.pt')
        inference.export_for_inference(
            model=model,
            save_format=save_format,
            save_path=save_path,
        )
        loaded_model = torch.jit.load(save_path)
        loaded_model.eval()
        loaded_model_out = loaded_model(sample_input)

        torch.testing.assert_close(
            orig_out,
            loaded_model_out,
            msg=f'output mismatch with {save_format}',
        )


@device('cpu', 'gpu')
@pytest.mark.parametrize('onnx_opset_version', [13, None])
def test_huggingface_export_for_inference_onnx(onnx_opset_version, tiny_bert_config, device):
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    pytest.importorskip('transformers')

    if onnx_opset_version == None and version.parse(torch.__version__) < version.parse('1.13'):
        pytest.skip("Don't test prior PyTorch version's default Opset version.")

    import onnx
    import onnx.checker
    import onnxruntime as ort
    import transformers

    from composer.models import HuggingFaceModel

    composer_device = get_device(device)
    cpu_device = get_device('cpu')

    # HuggingFace Bert Model
    # dummy sequence batch with 2 labels, 32 sequence length, and 30522 (bert) vocab size).
    input_ids = torch.randint(low=0, high=30522, size=(2, 32))
    labels = torch.randint(low=0, high=1, size=(2,))
    token_type_ids = torch.zeros(size=(2, 32), dtype=torch.int64)
    attention_mask = torch.randint(low=0, high=1, size=(2, 32))
    sample_input = {
        'input_ids': input_ids,
        'labels': labels,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
    }
    dynamic_axes = {
        'input_ids': {
            0: 'batch_size',
            1: 'seq_len'
        },
        'labels': {
            0: 'batch_size'
        },
        'token_type_ids': {
            0: 'batch_size',
            1: 'seq_len'
        },
        'attention_mask': {
            0: 'batch_size',
            1: 'seq_len'
        },
    }

    tiny_bert_config.num_labels = 2
    tiny_bert_config.hidden_act = 'gelu_new'
    hf_model = transformers.AutoModelForSequenceClassification.from_config(
        tiny_bert_config)  # type: ignore (thirdparty)

    model = HuggingFaceModel(hf_model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    apply_gated_linear_units(model, optimizer)

    model.eval()

    # Move model to device
    composer_device.module_to_device(model)
    for key, val in sample_input.items():
        sample_input[key] = composer_device.tensor_to_device(val)

    orig_out = model(sample_input)

    save_format = 'onnx'
    with tempfile.TemporaryDirectory() as tempdir:
        save_path = os.path.join(tempdir, f'model.{save_format}')
        inference.export_for_inference(
            model=model,
            save_format=save_format,
            save_path=save_path,
            sample_input=(sample_input, {}),
            dynamic_axes=dynamic_axes,
            onnx_opset_version=onnx_opset_version,
        )
        loaded_model = onnx.load(save_path)

        onnx.checker.check_model(loaded_model)

        ort_session = ort.InferenceSession(save_path)

        for key, value in sample_input.items():
            sample_input[key] = cpu_device.tensor_to_device(value).numpy()

        loaded_model_out = ort_session.run(None, sample_input)

        torch.testing.assert_close(
            cpu_device.tensor_to_device(orig_out['logits'].detach()).numpy(),
            loaded_model_out[1],
            rtol=1e-4,  # lower tolerance for ONNX
            atol=1e-3,  # lower tolerance for ONNX
            msg=f'output mismatch with {save_format}',
        )


@device('cpu', 'gpu')
@pytest.mark.parametrize(
    'model_cls, sample_input',
    [(partial(composer_resnet, 'resnet18'), (torch.rand(4, 3, 224, 224), torch.randint(10, (4,)))),
     (SimpleTransformerClassifier, dummy_transformer_classifier_batch())],
)
@pytest.mark.parametrize('onnx_opset_version', [13, None])
def test_export_for_inference_onnx(model_cls, sample_input, onnx_opset_version, device):
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')

    if onnx_opset_version == None and version.parse(torch.__version__) < version.parse('1.13'):
        pytest.skip("Don't test prior PyTorch version's default Opset version.")

    from composer.utils.misc import using_torch_2
    if using_torch_2():
        pytest.xfail(
            'torch.onnx.errors.UnsupportedOperatorError: Exporting the operator "aten::unflatten" to ONNX opset version 14 is not supported.'
        )

    import onnx
    import onnx.checker
    import onnxruntime as ort

    model = model_cls()
    model.eval()

    composer_device = get_device(device)
    cpu_device = get_device('cpu')
    sample_input = (composer_device.tensor_to_device(sample_input[0]),
                    composer_device.tensor_to_device(sample_input[1]))
    composer_device.module_to_device(model)
    orig_out = model(sample_input)
    save_format = 'onnx'
    with tempfile.TemporaryDirectory() as tempdir:
        save_path = os.path.join(tempdir, f'model.{save_format}')
        inference.export_for_inference(
            model=model,
            save_format=save_format,
            save_path=save_path,
            sample_input=(sample_input, {}),
            onnx_opset_version=onnx_opset_version,
        )
        loaded_model = onnx.load(save_path)
        onnx.checker.check_model(loaded_model)

        ort_session = ort.InferenceSession(save_path)
        loaded_model_out = ort_session.run(
            None,
            {'input': cpu_device.tensor_to_device(sample_input[0]).numpy()},
        )

        torch.testing.assert_close(
            cpu_device.tensor_to_device(orig_out.detach()).numpy(),
            loaded_model_out[0],
            rtol=1e-4 if isinstance(composer_device, DeviceCPU) else 1e-3,  # lower tolerance for ONNX
            atol=1e-3 if isinstance(composer_device, DeviceCPU) else 1e-2,  # lower tolerance for ONNX
            msg=lambda msg: f'output mismatch with {save_format}\n\nOriginal message: {msg}',
        )


@pytest.mark.parametrize(
    'model_cls, sample_input',
    [
        (partial(composer_resnet, 'resnet18'), (torch.rand(1, 3, 224, 224), torch.randint(10, (1,)))),
        (SimpleTransformerClassifier, dummy_transformer_classifier_batch()),
        pytest.param(configure_tiny_bert_hf_model,
                     dummy_tiny_bert_lm_batch(),
                     marks=pytest.mark.xfail(reason='HuggingFace models do not support torch.jit.script()')),
    ],
)
@pytest.mark.world_size(2)
def test_export_for_inference_torchscript_ddp(model_cls, sample_input, request: pytest.FixtureRequest):
    model = model_cls()
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    device = None
    for item in request.session.items:
        device = DeviceCPU() if item.get_closest_marker('gpu') is None else DeviceGPU()
        break
    assert device != None

    state = State(
        model=model,
        rank_zero_seed=0,
        device=device,
        run_name='run_name',
        optimizers=optimizer,
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
            inference.export_for_inference(
                model=state.model.module,
                save_format=save_format,
                save_path=save_path,
            )

            loaded_model = torch.jit.load(save_path)
            loaded_model.eval()
            loaded_model_out = loaded_model(sample_input)

            torch.testing.assert_close(orig_out, loaded_model_out)


@pytest.mark.parametrize(
    'model_cls, sample_input',
    [
        (partial(composer_resnet, 'resnet18'), (torch.rand(1, 3, 224, 224), torch.randint(10, (1,)))),
        (SimpleTransformerClassifier, dummy_transformer_classifier_batch()),
    ],
)
@pytest.mark.world_size(2)
@pytest.mark.parametrize('onnx_opset_version', [13, None])
def test_export_for_inference_onnx_ddp(model_cls, sample_input, onnx_opset_version, request: pytest.FixtureRequest):
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')

    from composer.utils.misc import using_torch_2
    if using_torch_2():
        pytest.xfail(
            'torch.onnx.errors.UnsupportedOperatorError: Exporting the operator "aten::unflatten" to ONNX opset version 14 is not supported.'
        )

    if onnx_opset_version == None and version.parse(torch.__version__) < version.parse('1.13'):
        pytest.skip("Don't test prior PyTorch version's default Opset version.")

    import onnx
    import onnx.checker
    import onnxruntime as ort

    model = model_cls()
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    device = None
    for item in request.session.items:
        device = DeviceCPU() if item.get_closest_marker('gpu') is None else DeviceGPU()
        break
    assert device != None

    state = State(
        model=model,
        rank_zero_seed=0,
        device=device,
        run_name='run_name',
        optimizers=optimizer,
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
            inference.export_for_inference(
                model=state.model.module,
                save_format=save_format,
                save_path=save_path,
                sample_input=(sample_input, {}),
                onnx_opset_version=onnx_opset_version,
            )

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
    'model_cls, dataloader',
    [(partial(composer_resnet, 'resnet18'), DataLoader(RandomImageDataset(shape=(3, 224, 224)))),
     (SimpleTransformerClassifier, dummy_text_classification_dataloader())],
)
def test_export_with_file_uploading_logger(model_cls, dataloader):
    with patch('composer.utils.inference.export_for_inference'):
        save_format = 'torchscript'
        model = model_cls()
        mock_obj_logger = MockFileUploader()
        with tempfile.TemporaryDirectory() as tempdir:
            save_path = os.path.join(tempdir, f'model.pt')

            # Construct the trainer and train
            trainer = Trainer(
                model=model,
                train_dataloader=dataloader,
                max_duration='1ba',
            )
            trainer.fit()

            mock_logger = Logger(state=trainer.state, destinations=[mock_obj_logger])

            export_with_logger(
                model=model,
                save_format=save_format,
                save_path=save_path,
                logger=mock_logger,
            )

            # Assert export_for_inference utility called with expected inputs
            inference.export_for_inference.assert_called_once_with(
                model=model,
                save_format=save_format,
                save_path=ANY,
                sample_input=ANY,
                transforms=None,
            )


@pytest.mark.parametrize(
    'model_cls, dataloader',
    [(partial(composer_resnet, 'resnet18'), DataLoader(RandomImageDataset(shape=(3, 224, 224)))),
     (SimpleTransformerClassifier, dummy_text_classification_dataloader())],
)
def test_export_with_other_logger(model_cls, dataloader):
    with patch('composer.utils.inference.export_for_inference'):
        save_format = 'torchscript'
        model = model_cls()
        non_file_uploading_logger = InMemoryLogger()
        with tempfile.TemporaryDirectory() as tempdir:
            save_path = os.path.join(tempdir, f'model.pt')

            # Construct the trainer and train
            trainer = Trainer(
                model=model,
                train_dataloader=dataloader,
                max_duration='1ba',
            )
            trainer.fit()

            mock_logger = Logger(
                state=trainer.state,
                destinations=[non_file_uploading_logger],
            )

            export_with_logger(
                model=model,
                save_format=save_format,
                save_path=save_path,
                logger=mock_logger,
            )

            # Assert export_for_inference utility called with expected inputs
            inference.export_for_inference.assert_called_once_with(
                model=model,
                save_format=save_format,
                save_path=save_path,
                save_object_store=None,
                sample_input=ANY,
                transforms=None,
            )


class LinModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 256)

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x


@pytest.mark.parametrize(
    'model_cls',
    [
        (LinModel),
    ],
)
def test_dynamic_quantize(model_cls):
    model = model_cls()

    save_format = 'torchscript'
    with tempfile.TemporaryDirectory() as tempdir:
        save_path_no_quantize = os.path.join(tempdir, f'model_no_quantize.pt')
        inference.export_for_inference(
            model=model,
            save_format=save_format,
            save_path=save_path_no_quantize,
        )
        save_path_quantize = os.path.join(tempdir, f'model_quantize.pt')
        inference.export_for_inference(
            model=model,
            save_format=save_format,
            save_path=save_path_quantize,
            transforms=[inference.quantize_dynamic],
        )
        no_quantize_size = os.path.getsize(save_path_no_quantize)
        quantize_size = os.path.getsize(save_path_quantize)
        # Size different should be almost 4x
        assert no_quantize_size > 3 * quantize_size, "Quantization didn't work"
