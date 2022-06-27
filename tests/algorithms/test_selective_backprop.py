# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
import torch

from composer.algorithms.selective_backprop import SelectiveBackprop
from composer.algorithms.selective_backprop.selective_backprop import select_using_loss, should_selective_backprop
from composer.core import Event
from composer.core.state import State
from composer.loggers import Logger
from composer.models import ComposerClassifier


@pytest.fixture
def N() -> int:
    """Batch size."""
    return 16


@pytest.fixture
def D() -> int:
    """Input dimension."""
    return 8


@pytest.fixture
def X(N: int, D: int) -> torch.Tensor:
    """2D input."""
    torch.manual_seed(42)
    return torch.randn(N, D)


@pytest.fixture
def X3D(N: int, D: int) -> torch.Tensor:
    """3D input."""
    torch.manual_seed(42)
    return torch.randn(N, D, D)


@pytest.fixture
def X5D(N: int, D: int) -> torch.Tensor:
    """5D input."""
    torch.manual_seed(42)
    return torch.randn(N, D, D, D, D)


@pytest.fixture
def Ximage(N: int) -> torch.Tensor:
    """4D image input."""
    torch.manual_seed(42)
    return torch.randn(N, 3, 32, 32)


@pytest.fixture
def y(N: int) -> torch.Tensor:
    """Target."""
    torch.manual_seed(42)
    return torch.randint(2, (N,))


@pytest.fixture
def loss_fun() -> Callable:
    """Fake loss function."""

    def loss(output, target, reduction='none'):
        return torch.ones_like(target)

    return loss


@pytest.fixture
def loss_fun_tuple() -> Callable:
    """Fake loss function that requires a batch tuple."""

    def loss(output, batch, reduction='none'):
        _, target = batch
        return torch.ones_like(target)

    return loss


@pytest.fixture
def bad_loss() -> Callable:
    """Fake loss function that will error."""

    def loss(output, target):
        return 0

    return loss


@pytest.fixture
def model(X: torch.Tensor) -> torch.nn.Module:
    """Simple fake linear model."""
    return torch.nn.Linear(X.shape[1], 1)


@pytest.fixture
def model3D(X3D: torch.Tensor) -> torch.nn.Module:
    """Simple fake linear model."""
    return torch.nn.Sequential(torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(), torch.nn.Linear(X3D.shape[1], 1))


@pytest.fixture
def model5D(X5D: torch.Tensor) -> torch.nn.Module:
    """Simple fake linear model."""
    return torch.nn.Sequential(torch.nn.AdaptiveAvgPool3d(1), torch.nn.Flatten(), torch.nn.Linear(X5D.shape[1], 1))


@pytest.fixture
def keep() -> float:
    """keep hparam."""
    return 0.5


@pytest.fixture
def scale_factor() -> float:
    """scale_factor hparam."""
    return 0.5


@pytest.fixture
def epoch() -> int:
    """Default epoch."""
    return 5


@pytest.fixture
def batch() -> int:
    """Default batch."""
    return 0


@pytest.fixture
def conv_model(Ximage: torch.Tensor, D: int) -> ComposerClassifier:
    """Dummy conv model."""
    return ComposerClassifier(torch.nn.Conv2d(Ximage.shape[1], D, 3))


@pytest.fixture
def state(minimal_state: State, conv_model: ComposerClassifier, loss_fun_tuple: Callable, epoch: int,
          batch: int) -> State:
    """State with required values set for Selective Backprop."""
    assert minimal_state.dataloader_len is not None
    conv_model.loss = loss_fun_tuple
    minimal_state.model = conv_model

    minimal_state.timestamp = minimal_state.timestamp.copy(
        epoch=epoch,
        batch=epoch * int(minimal_state.dataloader_len) + batch,
        batch_in_epoch=batch,
    )

    return minimal_state


# tests of the functional API
class TestSelectiveBackprop:

    @pytest.mark.parametrize('epoch,batch,interrupt', [(10, 0, 0), (10, 0, 2), (10, 2, 2)])
    def test_select_using_loss_true(self, epoch: int, batch: int, interrupt: int) -> None:
        """Test functional match when epoch is within interval."""
        start = 5
        end = 15
        is_chosen = should_selective_backprop(epoch, batch, start, end, interrupt)
        assert is_chosen

    @pytest.mark.parametrize('epoch,batch,interrupt', [(0, 0, 0), (20, 0, 0), (10, 1, 2)])
    def test_select_using_loss_false(self, epoch: int, batch: int, interrupt: int) -> None:
        """Test functional doesn't match when epoch is outside of interval."""
        start = 5
        end = 15
        is_chosen = should_selective_backprop(epoch, batch, start, end, interrupt)
        assert not is_chosen

    @pytest.mark.parametrize('keep', [0.5])
    @pytest.mark.parametrize('scale_factor', [0.5])
    @pytest.mark.xfail()
    def test_selective_output_shape_3D(self, X3D: torch.Tensor, y: torch.Tensor, model: torch.nn.Module,
                                       loss_fun: Callable, keep: float, scale_factor: float) -> None:
        """Test functional selection on 3D inputs."""
        N, D, _ = X3D.shape

        X_scaled, y_scaled = select_using_loss(X3D, y, model, loss_fun, keep, scale_factor)
        assert X_scaled.shape == (int(N * keep), D, D)
        assert y_scaled.shape == (int(N * keep),)

    @pytest.mark.parametrize('keep', [1, 0.5, 0.75])
    @pytest.mark.parametrize('scale_factor', [1])
    def test_selective_output_shape(self, X: torch.Tensor, y: torch.Tensor, model: torch.nn.Module, loss_fun: Callable,
                                    keep: float, scale_factor: float) -> None:
        """Test functional selection on 2D inputs."""
        N, D = X.shape

        X_scaled, y_scaled = select_using_loss(X, y, model, loss_fun, keep, scale_factor)
        assert X_scaled.shape == (int(N * keep), D)
        assert y_scaled.shape == (int(N * keep),)

    @pytest.mark.parametrize('keep', [0.5, 0.75, 1])
    @pytest.mark.parametrize('scale_factor', [0.5, 0.75])
    def test_selective_output_shape_scaled(self, Ximage: torch.Tensor, y: torch.Tensor, conv_model: ComposerClassifier,
                                           loss_fun: Callable, keep: float, scale_factor: float) -> None:
        """Test functional selection on 4D inputs."""
        N, C, H, W = Ximage.shape
        X_scaled, y_scaled = select_using_loss(Ximage, y, conv_model.module, loss_fun, keep, scale_factor)
        assert X_scaled.shape == (int(N * keep), C, H, W)
        assert y_scaled.shape == (int(N * keep),)

    def test_selective_backprop_interp_dim_error(self, X: torch.Tensor, y: torch.Tensor, model: torch.nn.Module,
                                                 loss_fun: Callable) -> None:
        """Ensure that ValueError is raised when input tensor can't be scaled."""
        with pytest.raises(ValueError):
            select_using_loss(X, y, model, loss_fun, 1, 0.5)

    def test_selective_backprop_bad_loss_error(self, X: torch.Tensor, y: torch.Tensor, model: torch.nn.Module,
                                               bad_loss: Callable) -> None:
        """Ensure that ValueError is raised when loss function doesn't have `reduction` kwarg."""
        with pytest.raises(TypeError) as execinfo:
            select_using_loss(X, y, model, bad_loss, 1, 1)
        MATCH = 'must take a keyword argument `reduction`.'
        assert MATCH in str(execinfo.value)


class TestSelectiveBackpropAlgorithm:
    """
    Test Selective Backprop Algorithm
    """

    @pytest.fixture
    def sb_algorithm(self, scale_factor, keep) -> SelectiveBackprop:
        return SelectiveBackprop(
            start=0.5,
            end=0.8,
            keep=keep,
            scale_factor=scale_factor,
            interrupt=2,
        )

    @pytest.mark.parametrize('event', [Event.AFTER_DATALOADER])
    @pytest.mark.parametrize('epoch,batch', [(5, 0), (7, 0), (5, 2)])
    def test_match_correct(self, event: Event, sb_algorithm: SelectiveBackprop, state: State) -> None:
        """Algo should match AFTER_DATALOADER in the right interval."""
        state.max_duration = '10ep'

        assert sb_algorithm.match(event, state)

    @pytest.mark.parametrize('event,epoch,batch', [(Event.AFTER_DATALOADER, 0, 0), (Event.AFTER_DATALOADER, 5, 1)])
    def test_match_incorrect(self, event: Event, sb_algorithm: SelectiveBackprop, state: State) -> None:
        """Algo should NOT match the wrong interval."""
        state.max_duration = '10ep'

        assert not sb_algorithm.match(event, state)

    @pytest.mark.parametrize('epoch,batch', [(5, 0)])
    @pytest.mark.parametrize('keep', [0.5, 0.75, 1])
    @pytest.mark.parametrize('scale_factor', [0.5, 1])
    def test_apply(self, Ximage: torch.Tensor, y: torch.Tensor, sb_algorithm: SelectiveBackprop, state: State,
                   empty_logger: Logger, keep: float) -> None:
        """Test apply with image inputs gives the right output shape."""
        N, C, H, W = Ximage.shape

        state.max_duration = '10ep'
        state.batch = (Ximage, y)
        sb_algorithm.apply(Event.INIT, state, empty_logger)
        sb_algorithm.apply(Event.AFTER_DATALOADER, state, empty_logger)

        X_scaled, y_scaled = state.batch
        assert X_scaled.shape == (int(N * keep), C, H, W)
        assert y_scaled.shape == (int(N * keep),)
