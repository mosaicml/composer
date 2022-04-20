import pytest
import torch
from torchvision.models import resnet50

from composer.functional import (apply_blurpool, apply_channels_last, apply_factorization, apply_ghost_batchnorm,
                                 apply_squeeze_excite, apply_stochastic_depth)

algo_kwargs = {apply_stochastic_depth: {'stochastic_method': 'block', 'target_layer_name': 'ResNetBottleneck'}}


@pytest.fixture
def input():
    return torch.Tensor(1, 3, 224, 224)


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
    model.train()

    scripted_func = torch.jit.script(model)
    torch.testing.assert_allclose(scripted_func(input), model(input))  # type: ignore (third-party)

    scripted_func(input)


@pytest.mark.parametrize("surgery_method", [
    pytest.param(apply_blurpool),
    pytest.param(apply_factorization, marks=pytest.mark.xfail),
    pytest.param(apply_ghost_batchnorm, marks=pytest.mark.xfail),
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
    model.eval()

    scripted_func = torch.jit.script(model)
    torch.testing.assert_allclose(scripted_func(input), model(input))  # type: ignore (third-party)
