# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from composer.algorithms.progressive_resizing import ProgressiveResizing, resize_batch
from composer.core import Event
from composer.core.state import State
from composer.core.time import TimeUnit
from composer.loggers import Logger


def check_scaled_shape(orig: torch.Tensor, scaled: torch.Tensor, scale_factor: float) -> bool:
    """Asserts that the scaled shape is correct, given orig shape and scale_factor."""
    N, C, H, W = orig.shape
    Hc = int(scale_factor * H)
    Wc = int(scale_factor * W)

    return scaled.shape == (N, C, Hc, Wc)


@pytest.fixture
def Wx() -> int:
    return 32


@pytest.fixture
def Hx(Wx: int) -> int:
    return Wx


@pytest.fixture
def X(Wx: int, Hx: int):
    torch.manual_seed(0)
    return torch.randn(16, 8, Hx, Wx)


@pytest.fixture
def Wy(Wx: int) -> int:
    return Wx


@pytest.fixture
def Hy(Hx: int) -> int:
    return Hx


@pytest.fixture
def y(Wy: int, Hy: int):
    torch.manual_seed(0)
    return torch.randn(16, 8, Hy, Wy)


@pytest.fixture(params=[0.5, 0.75, 1])
def scale_factor(request) -> float:
    return request.param


@pytest.fixture(params=['resize', 'crop'])
def mode(request) -> str:
    return request.param


@pytest.fixture
def initial_scale() -> float:
    return 0.5


@pytest.fixture
def finetune_fraction() -> float:
    return 0.2


@pytest.fixture
def delay_fraction() -> float:
    return 0.2


@pytest.fixture
def size_increment() -> float:
    return 8


@pytest.fixture
def resize_targets() -> bool:
    return False


class TestResizeInputs:

    def test_resize_noop(self, X, y, mode):
        """Tests that no operation is performed when scale_factor == 1."""
        Xc, _ = resize_batch(X, y, 1.0, mode, resize_targets=False)
        assert X is Xc

    @pytest.mark.parametrize('y', [None])
    def test_without_target(self, X, y):
        """Test that resizing works properly with no target present."""
        try:
            resize_batch(X, y, 1.0, 'crop', resize_targets=False)
        except:
            pytest.fail('apply_progressive_resizing failed with y == None')

    @pytest.mark.parametrize('Wx,Hx', [(31, 31), (32, 32), (32, 16)])
    def test_resize_batch_shape(self, X: torch.Tensor, y: torch.Tensor, mode: str, scale_factor: float):
        """Test scaling works for different input shapes."""
        Xc, _ = resize_batch(X, y, scale_factor, mode, resize_targets=False)
        assert check_scaled_shape(X, Xc, scale_factor)

    def test_resize_outputs_shape(self, X: torch.Tensor, y: torch.Tensor, mode: str, scale_factor: float):
        """Test that resizing outputs works."""
        _, yc = resize_batch(X, y, scale_factor, mode, resize_targets=True)
        assert check_scaled_shape(y, yc, scale_factor)

    def test_resize_outputs_crop(self, X: torch.Tensor, scale_factor: float):
        """Test that resizing outputs in crop mode gives the right targets."""
        xc, yc = resize_batch(X, X, scale_factor, 'crop', resize_targets=True)
        assert torch.equal(xc, yc)

    @pytest.mark.parametrize('Wx,Hx,Wy,Hy', [(32, 32, 16, 16)])
    def test_resize_outputs_different_shape(self, X, y, scale_factor: float, mode: str):
        """Test that resizing works when X and y have different shapes."""
        _, yc = resize_batch(X, y, scale_factor, mode, resize_targets=True)
        assert check_scaled_shape(y, yc, scale_factor)


@pytest.mark.parametrize('mode,initial_scale,finetune_fraction,delay_fraction,size_increment',
                         [('foo', 0.5, 0.2, 0.2, 8), ('crop', 1.2, 0.2, 0.2, 8), ('crop', 0.5, 1.2, 0.2, 8),
                          ('resize', 0.5, 0.6, 0.5, 8)])
def test_invalid_hparams(mode: str, initial_scale: float, finetune_fraction: float, delay_fraction: float,
                         size_increment: int):
    """Test that invalid hyperparameters error.

    Ideally this could be caught by the Hparams, but that's not yet supported in yahp.
    """
    with pytest.raises(ValueError):
        ProgressiveResizing(mode, initial_scale, finetune_fraction, delay_fraction, size_increment, False)


class TestProgressiveResizingAlgorithm:

    @pytest.fixture
    def pr_algorithm(self, mode, initial_scale, finetune_fraction, delay_fraction, size_increment, resize_targets):
        return ProgressiveResizing(mode, initial_scale, finetune_fraction, delay_fraction, size_increment,
                                   resize_targets)

    @pytest.mark.parametrize('event', [Event.AFTER_DATALOADER])
    def test_match_correct(self, event: Event, pr_algorithm, minimal_state: State):
        """Algo should match AFTER_DATALOADER."""
        assert pr_algorithm.match(event, minimal_state)

    @pytest.mark.parametrize('event', [Event.INIT])
    def test_match_incorrect(self, event: Event, pr_algorithm: ProgressiveResizing, minimal_state: State):
        """Algo should NOT match INIT."""
        assert not pr_algorithm.match(event, minimal_state)

    @pytest.mark.parametrize('epoch_frac', [0.0, 0.6, 0.8, 1.0])
    def test_apply(self, epoch_frac: float, X: torch.Tensor, y: torch.Tensor, pr_algorithm: ProgressiveResizing,
                   minimal_state: State, empty_logger: Logger):
        """Test apply at different epoch fractions (fraction of max epochs)"""
        assert minimal_state.max_duration is not None
        assert minimal_state.max_duration.unit == TimeUnit.EPOCH
        minimal_state.timestamp.epoch._value = int(epoch_frac * minimal_state.max_duration.value)
        s = pr_algorithm.initial_scale
        f = pr_algorithm.finetune_fraction
        d = pr_algorithm.delay_fraction
        p = pr_algorithm.size_increment
        if epoch_frac >= d:
            scale_factor_elapsed = min([(epoch_frac - d) / (1 - f - d), 1.0])
        else:
            scale_factor_elapsed = 0.0
        scale_factor = s + (1 - s) * scale_factor_elapsed
        minimal_state.batch = (X, y)
        width = X.shape[3]
        scaled_width_pinned = round(width * scale_factor / p) * p
        scale_factor_pinned = scaled_width_pinned / width

        pr_algorithm.apply(Event.AFTER_DATALOADER, minimal_state, empty_logger)

        last_input, _ = minimal_state.batch
        assert check_scaled_shape(X, last_input, scale_factor_pinned)
