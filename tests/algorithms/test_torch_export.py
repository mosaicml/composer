# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""
Tests a variety of export options with our surgery methods applied, including
torchscript, torch.fx, and ONNX.
"""
import os
import pathlib
from typing import Any, Callable, Type

import pytest
import torch
import torch.fx

from composer.algorithms.blurpool.blurpool import BlurPool
from composer.algorithms.channels_last.channels_last import ChannelsLast
from composer.algorithms.factorize.factorize import Factorize
from composer.algorithms.ghost_batchnorm.ghost_batchnorm import GhostBatchNorm
from composer.algorithms.squeeze_excite.squeeze_excite import SqueezeExcite
from composer.algorithms.stochastic_depth.stochastic_depth import StochasticDepth
from composer.core.algorithm import Algorithm
from composer.functional import (apply_blurpool, apply_channels_last, apply_factorization, apply_ghost_batchnorm,
                                 apply_squeeze_excite, apply_stochastic_depth)
from tests.algorithms.algorithm_settings import get_alg_kwargs, get_alg_model, get_algs_with_marks

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


torchscript_algs_with_marks = [
    x for x in get_algs_with_marks()
    if x.values[0] in (BlurPool, Factorize, GhostBatchNorm, SqueezeExcite, StochasticDepth, ChannelsLast)
]

# <--- torchscript export --->


def get_surgery_method(alg_cls: Type[Algorithm]) -> Callable:
    if alg_cls is BlurPool:
        return apply_blurpool
    if alg_cls is Factorize:
        return apply_factorization
    if alg_cls is GhostBatchNorm:
        return apply_ghost_batchnorm
    if alg_cls is SqueezeExcite:
        return apply_squeeze_excite
    if alg_cls is StochasticDepth:
        return apply_stochastic_depth
    if alg_cls is ChannelsLast:
        return apply_channels_last
    raise ValueError(f'Unknown algorithm class {alg_cls}')


@pytest.mark.parametrize('alg_cls', torchscript_algs_with_marks)
def test_surgery_torchscript_train(input: Any, alg_cls: Type[Algorithm]):
    """Tests torchscript model in train mode."""
    if alg_cls in (Factorize, GhostBatchNorm, StochasticDepth):
        pytest.xfail('Unsupported')

    alg_kwargs = get_alg_kwargs(alg_cls)
    model = get_alg_model(alg_cls)

    surgery_method = get_surgery_method(alg_cls)

    alg_kwargs = algo_kwargs.get(surgery_method, alg_kwargs)

    surgery_method(model, **alg_kwargs)

    scripted_func = torch.jit.script(model)
    scripted_func.train()  # type: ignore (third-party)
    model.train()
    torch.testing.assert_close(scripted_func(input), model(input))  # type: ignore (third-party)


@pytest.mark.parametrize('alg_cls', torchscript_algs_with_marks)
def test_surgery_torchscript_eval(input: Any, alg_cls: Type[Algorithm]):
    """Tests torchscript model in eval mode."""
    if alg_cls is Factorize:
        pytest.xfail('Unsupported')

    surgery_method = get_surgery_method(alg_cls)

    alg_kwargs = get_alg_kwargs(alg_cls)
    model = get_alg_model(alg_cls)
    alg_kwargs = algo_kwargs.get(surgery_method, alg_kwargs)

    surgery_method(model, **alg_kwargs)

    scripted_func = torch.jit.script(model)
    scripted_func.eval()  # type: ignore (third-party)
    model.eval()
    torch.testing.assert_close(scripted_func(input), model(input))  # type: ignore (third-party)


# <--- torch.fx export --->


@pytest.mark.parametrize('alg_cls', torchscript_algs_with_marks)
def test_surgery_torchfx_eval(
    input: Any,
    alg_cls: Type[Algorithm],
):
    """Tests torch.fx model in eval mode."""

    alg_kwargs = get_alg_kwargs(alg_cls)
    model = get_alg_model(alg_cls)
    surgery_method = get_surgery_method(alg_cls)

    if alg_cls in (BlurPool, GhostBatchNorm):
        pytest.xfail('Control flow')

    alg_kwargs = algo_kwargs.get(surgery_method, alg_kwargs)

    surgery_method(model, **alg_kwargs)

    model.eval()

    traced_func = torch.fx.symbolic_trace(model)
    torch.testing.assert_close(traced_func(input), model(input))  # type: ignore (third-party)


# <--- onnx export --->


@pytest.mark.parametrize('alg_cls', torchscript_algs_with_marks)
@pytest.mark.filterwarnings(
    r'ignore:Converting a tensor to a Python .* might cause the trace to be incorrect:torch.jit._trace.TracerWarning')
def test_surgery_onnx(
    input: Any,
    alg_cls: Type[Algorithm],
    tmp_path: pathlib.Path,
):
    """Tests onnx export and runtime"""
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    import onnx
    import onnxruntime as ort

    surgery_method = get_surgery_method(alg_cls)

    model = get_alg_model(alg_cls)
    alg_kwargs = get_alg_kwargs(alg_cls)
    alg_kwargs = algo_kwargs.get(surgery_method, alg_kwargs)

    surgery_method(model, **alg_kwargs)
    model.eval()

    onnx_path = os.path.join(tmp_path, 'model.onnx')
    torch.onnx.export(
        model,
        (input,),
        onnx_path,
        input_names=['input'],
        output_names=['output'],
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

    torch.testing.assert_close(
        outputs[0],
        model(input).detach().numpy(),
        rtol=1e-4,  # lower tolerance for ONNX
        atol=1e-3,  # lower tolerance for ONNX
    )
