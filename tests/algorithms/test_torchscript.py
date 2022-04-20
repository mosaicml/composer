from composer.functional import (apply_blurpool, apply_factorization, apply_ghost_batchnorm, apply_squeeze_excite)
import torch
from torchvision.models import densenet161
import pytest


@pytest.mark.parametrize("surgery_method", [
    pytest.param(apply_blurpool),
    pytest.param(apply_factorization, marks=pytest.mark.xfail),
    pytest.param(apply_ghost_batchnorm, marks=pytest.mark.xfail),
    pytest.param(apply_squeeze_excite),
])
def test_surgery_torchscript(surgery_method):

    model = densenet161()
    model = surgery_method(model)
    torch.jit.script(model)
