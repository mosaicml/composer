# Copyright 2021 MosaicML. All Rights Reserved.

import pytest
import torch

from composer.algorithms.progressive_resizing import ProgressiveResizing, ProgressiveResizingHparams, resize_inputs
from composer.core import Event
from composer.trainer import TrainerHparams
from tests.utils.trainer_fit import train_model


def check_scaled_shape(orig: torch.Tensor, scaled: torch.Tensor, scale_factor: float) -> bool:
    """ Asserts that the scaled shape is correct, given orig shape and scale_factor"""
    N, C, H, W = orig.shape
    Hc = int(scale_factor * H)
    Wc = int(scale_factor * W)

    return scaled.shape == (N, C, Hc, Wc)


@pytest.fixture
def Wx() -> int:
    return 32


@pytest.fixture
def Hx(Wx) -> int:
    return Wx


@pytest.fixture
def X(Wx, Hx):
    torch.manual_seed(0)
    return torch.randn(16, 8, Hx, Wx)


@pytest.fixture
def Wy(Wx) -> int:
    return Wx


@pytest.fixture
def Hy(Hx) -> int:
    return Hx


@pytest.fixture
def y(Wy, Hy):
    torch.manual_seed(0)
    return torch.randn(16, 8, Hy, Wy)


@pytest.fixture(params=[0.5, 0.75, 1])
def scale_factor(request) -> float:
    return request.param


@pytest.fixture(params=["resize", "crop"])
def mode(request) -> str:
    return request.param


@pytest.fixture
def initial_scale() -> float:
    return 0.5


@pytest.fixture
def finetune_fraction() -> float:
    return 0.2


@pytest.fixture
def resize_targets() -> bool:
    return False


@pytest.fixture
def dummy_hparams(mode, initial_scale, finetune_fraction, resize_targets):
    return ProgressiveResizingHparams(mode, initial_scale, finetune_fraction, resize_targets)


@pytest.fixture
def dummy_algorithm(dummy_hparams):
    return ProgressiveResizing(dummy_hparams.mode, dummy_hparams.initial_scale, dummy_hparams.finetune_fraction,
                               dummy_hparams.resize_targets)


def test_resize_noop(X, y, mode):
    """ Tests that no operation is performed when scale_factor == 1"""
    Xc, yc = resize_inputs(X, y, 1.0, mode, resize_targets=False)
    assert X is Xc


@pytest.mark.parametrize("y", [None])
def test_without_target(X, y):
    """ Test that resizing works properly with no target present"""
    try:
        Xc, yc = resize_inputs(X, y, 1.0, "crop", resize_targets=False)
    except:
        pytest.fail("apply_progressive_resizing failed with y == None")


@pytest.mark.parametrize("Wx,Hx", [(31, 31), (32, 32), (32, 16)])
def test_resize_inputs_shape(X, y, mode, scale_factor):
    """ Test scaling works for different input shapes"""

    Xc, yc = resize_inputs(X, y, scale_factor, mode, resize_targets=False)
    assert check_scaled_shape(X, Xc, scale_factor)


def test_resize_outputs_shape(X, y, mode, scale_factor):
    """ Test that resizing outputs works """

    Xc, yc = resize_inputs(X, y, scale_factor, mode, resize_targets=True)
    assert check_scaled_shape(y, yc, scale_factor)


@pytest.mark.parametrize("Wx,Hx,Wy,Hy", [(32, 32, 16, 16)])
def test_resize_outputs_different_shape(X, y, scale_factor, mode):
    """ Test that resizing works when X and y have different shapes"""

    Xc, yc = resize_inputs(X, y, scale_factor, mode, resize_targets=True)
    assert check_scaled_shape(y, yc, scale_factor)


@pytest.mark.parametrize("mode,initial_scale,finetune_fraction", [("foo", 0.5, 0.2), ("crop", 1.2, 0.2),
                                                                  ("crop", 0.5, 1.2)])
def test_invalid_hparams(mode, initial_scale, finetune_fraction):
    """ Test that invalid hyperparameters error.
    Ideally this could be caught by the Hparams, but that's not yet supported in yahp.
    """
    with pytest.raises(ValueError):
        ProgressiveResizing(mode, initial_scale, finetune_fraction, False)


@pytest.mark.parametrize("event", [Event.AFTER_DATALOADER])
def test_match_correct(event, dummy_algorithm, dummy_state):
    """ Algo should match AFTER_DATALOADER """
    assert dummy_algorithm.match(event, dummy_state)


@pytest.mark.parametrize("event", [Event.TRAINING_START])
def test_match_incorrect(event, dummy_algorithm, dummy_state):
    """ Algo should NOT match TRAINING_START """
    assert not dummy_algorithm.match(event, dummy_state)


@pytest.mark.parametrize("epoch_frac", [0, 0.8, 1])
def test_apply(epoch_frac, X, y, dummy_algorithm, dummy_state, dummy_logger):
    """ Test apply at different epoch fractions (fraction of max epochs) """
    dummy_state.epoch = int(epoch_frac * dummy_state.max_epochs)
    s = dummy_algorithm.hparams.initial_scale
    f = dummy_algorithm.hparams.finetune_fraction
    scale_factor = min([s + (1 - s) / (1 - f) * epoch_frac, 1.0])
    dummy_state.batch = (X, y)
    dummy_algorithm.apply(Event.AFTER_DATALOADER, dummy_state, dummy_logger)

    last_input, _ = dummy_state.batch
    assert check_scaled_shape(X, last_input, scale_factor)


@pytest.mark.run_long
@pytest.mark.timeout(90)
def test_progressive_resizing_trains(mosaic_trainer_hparams: TrainerHparams):
    mosaic_trainer_hparams.algorithms = [
        ProgressiveResizingHparams(mode="resize", initial_scale=0.5, finetune_fraction=0.2, resize_targets=False)
    ]
    train_model(mosaic_trainer_hparams)
