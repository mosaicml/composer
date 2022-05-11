"""
Tests a variety of export options with our surgery methods applied, including
torchscript, torch.fx, and ONNX.
"""
import os

import pytest
import torch
import torch.fx
from torchvision.models import resnet50

from composer.functional import (apply_blurpool, apply_channels_last, apply_factorization, apply_ghost_batchnorm,
                                 apply_squeeze_excite, apply_stochastic_depth)

algo_kwargs = {
    apply_stochastic_depth: {
        'stochastic_method': 'block',
        'target_layer_name': 'ResNetBottleneck'
    },
    apply_ghost_batchnorm: {
        'ghost_batch_size': 2
    }
}


@pytest.fixture
def input():
    return torch.Tensor(4, 3, 224, 224)


# <--- torchscript export --->


@pytest.mark.parametrize("surgery_method", [
    pytest.param(apply_blurpool),
    pytest.param(apply_factorization, marks=pytest.mark.xfail),
    pytest.param(apply_ghost_batchnorm, marks=pytest.mark.xfail),
    pytest.param(apply_squeeze_excite),
    pytest.param(apply_stochastic_depth, marks=pytest.mark.xfail),
    pytest.param(apply_channels_last)
])
@pytest.mark.timeout(5)
def test_surgery_torchscript_train(surgery_method, input):
    """Tests torchscript model in train mode."""
    model = resnet50()
    kwargs = algo_kwargs.get(surgery_method, {})

    surgery_method(model, **kwargs)

    scripted_func = torch.jit.script(model)
    scripted_func.train()  # type: ignore (third-party)
    model.train()
    torch.testing.assert_allclose(scripted_func(input), model(input))  # type: ignore (third-party)


@pytest.mark.parametrize("surgery_method", [
    pytest.param(apply_blurpool),
    pytest.param(apply_factorization, marks=pytest.mark.xfail),
    pytest.param(apply_ghost_batchnorm),
    pytest.param(apply_squeeze_excite),
    pytest.param(apply_stochastic_depth),
    pytest.param(apply_channels_last)
])
@pytest.mark.timeout(5)
def test_surgery_torchscript_eval(surgery_method, input):
    """Tests torchscript model in eval mode."""
    model = resnet50()
    kwargs = algo_kwargs.get(surgery_method, {})

    surgery_method(model, **kwargs)

    scripted_func = torch.jit.script(model)
    scripted_func.eval()  # type: ignore (third-party)
    model.eval()
    torch.testing.assert_allclose(scripted_func(input), model(input))  # type: ignore (third-party)


# <--- torch.fx export --->


@pytest.mark.parametrize("surgery_method", [
    pytest.param(apply_blurpool, marks=pytest.mark.xfail(reason="control flow")),
    pytest.param(apply_factorization),
    pytest.param(apply_ghost_batchnorm, marks=pytest.mark.xfail(reason="control flow")),
    pytest.param(apply_squeeze_excite),
    pytest.param(apply_stochastic_depth),
    pytest.param(apply_channels_last)
])
@pytest.mark.timeout(5)
def test_surgery_torchfx_eval(surgery_method, input):
    """Tests torch.fx model in eval mode."""
    model = resnet50()
    kwargs = algo_kwargs.get(surgery_method, {})

    surgery_method(model, **kwargs)

    model.eval()

    traced_func = torch.fx.symbolic_trace(model)
    torch.testing.assert_allclose(traced_func(input), model(input))  # type: ignore (third-party)


# <--- onnx export --->


@pytest.mark.parametrize("surgery_method", [
    pytest.param(apply_blurpool),
    pytest.param(apply_factorization, marks=pytest.mark.xfail),
    pytest.param(apply_ghost_batchnorm),
    pytest.param(apply_squeeze_excite),
    pytest.param(apply_stochastic_depth),
    pytest.param(apply_channels_last)
])
@pytest.mark.timeout(5)
def test_surgery_onnx(surgery_method, input, tmpdir):
    """Tests onnx export and runtime"""
    pytest.importorskip("onnx")
    import onnx
    import onnxruntime as ort

    model = resnet50()
    kwargs = algo_kwargs.get(surgery_method, {})

    surgery_method(model, **kwargs)
    model.eval()

    onnx_path = os.path.join(tmpdir, "model.onnx")
    torch.onnx.export(
        model,
        input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
    )

    # check onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # run inference
    ort_session = ort.InferenceSession(onnx_path)
    outputs = ort_session.run(None, {'input': input})

    torch.testing.assert_allclose(outputs, model(input))  # type: ignore (third-party)