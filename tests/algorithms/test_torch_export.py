"""
Tests a variety of export options with our surgery methods applied, including
torchscript, torch.fx, and ONNX.
"""
import os

import pytest
import torch
import torch.fx

from composer.functional import (apply_blurpool, apply_channels_last, apply_factorization, apply_ghost_batchnorm,
                                 apply_squeeze_excite, apply_stochastic_depth)
from tests.algorithms import get_settings

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
    # input batch to ComposerModel is (input, target) tuple
    return (torch.rand(4, 3, 112, 112), torch.Tensor())


def get_model_and_algo_kwargs(name: str):
    settings = get_settings(name)
    assert settings is not None

    return settings["model"], settings["algorithm_kwargs"]


# <--- torchscript export --->


@pytest.mark.parametrize("name,surgery_method", [
    pytest.param("blurpool", apply_blurpool),
    pytest.param("factorize", apply_factorization, marks=pytest.mark.xfail),
    pytest.param("ghost_batchnorm", apply_ghost_batchnorm, marks=pytest.mark.xfail),
    pytest.param("squeeze_excite", apply_squeeze_excite),
    pytest.param("stochastic_depth", apply_stochastic_depth, marks=pytest.mark.xfail),
    pytest.param("channels_last", apply_channels_last)
])
@pytest.mark.timeout(10)
def test_surgery_torchscript_train(name, surgery_method, input):
    """Tests torchscript model in train mode."""

    model, kwargs = get_model_and_algo_kwargs(name)

    surgery_method(model, **kwargs)

    scripted_func = torch.jit.script(model)
    scripted_func.train()  # type: ignore (third-party)
    model.train()
    torch.testing.assert_allclose(scripted_func(input), model(input))  # type: ignore (third-party)


@pytest.mark.parametrize("name,surgery_method", [
    pytest.param("blurpool", apply_blurpool),
    pytest.param("factorize", apply_factorization, marks=pytest.mark.xfail),
    pytest.param("ghost_batchnorm", apply_ghost_batchnorm),
    pytest.param("squeeze_excite", apply_squeeze_excite),
    pytest.param("stochastic_depth", apply_stochastic_depth),
    pytest.param("channels_last", apply_channels_last)
])
@pytest.mark.timeout(10)
def test_surgery_torchscript_eval(name, surgery_method, input):
    """Tests torchscript model in eval mode."""

    model, kwargs = get_model_and_algo_kwargs(name)

    surgery_method(model, **kwargs)

    scripted_func = torch.jit.script(model)
    scripted_func.eval()  # type: ignore (third-party)
    model.eval()
    torch.testing.assert_allclose(scripted_func(input), model(input))  # type: ignore (third-party)


# <--- torch.fx export --->


@pytest.mark.parametrize("name,surgery_method", [
    pytest.param("blurpool", apply_blurpool, marks=pytest.mark.xfail(reason="control flow")),
    pytest.param("factorize", apply_factorization),
    pytest.param("ghost_batchnorm", apply_ghost_batchnorm, marks=pytest.mark.xfail(reason="control flow")),
    pytest.param("squeeze_excite", apply_squeeze_excite),
    pytest.param("stochastic_depth", apply_stochastic_depth),
    pytest.param("channels_last", apply_channels_last)
])
@pytest.mark.timeout(10)
def test_surgery_torchfx_eval(name, surgery_method, input):
    """Tests torch.fx model in eval mode."""

    model, kwargs = get_model_and_algo_kwargs(name)

    surgery_method(model, **kwargs)

    model.eval()

    traced_func = torch.fx.symbolic_trace(model)
    torch.testing.assert_allclose(traced_func(input), model(input))  # type: ignore (third-party)


# <--- onnx export --->


@pytest.mark.parametrize("name,surgery_method", [
    pytest.param("blurpool", apply_blurpool),
    pytest.param("factorize", apply_factorization),
    pytest.param("ghost_batchnorm", apply_ghost_batchnorm),
    pytest.param("squeeze_excite", apply_squeeze_excite),
    pytest.param("stochastic_depth", apply_stochastic_depth),
    pytest.param("channels_last", apply_channels_last)
])
@pytest.mark.timeout(10)
def test_surgery_onnx(name, surgery_method, input, tmpdir):
    """Tests onnx export and runtime"""
    pytest.importorskip("onnx")
    import onnx  # type: ignore
    import onnxruntime as ort  # type: ignore

    model, kwargs = get_model_and_algo_kwargs(name)

    surgery_method(model, **kwargs)
    model.eval()

    onnx_path = os.path.join(tmpdir, "model.onnx")
    torch.onnx.export(
        model,
        (input,),
        onnx_path,
        input_names=["input"],
        output_names=["output"],
    )

    # check onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # run inference
    ort_session = ort.InferenceSession(onnx_path)
    outputs = ort_session.run(
        None,
        {'input': input[0].numpy()},
    )

    torch.testing.assert_allclose(
        outputs[0],
        model(input),
        rtol=1e-4,  # lower tolerance for ONNX
        atol=1e-3,  # lower tolerance for ONNX
    )  # type: ignore (third-party)
