# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Callable

import pytest
import torch

from composer.algorithms.selective_backprop import SelectiveBackprop, SelectiveBackpropHparams
from composer.algorithms.selective_backprop.selective_backprop import do_selective_backprop, selective_backprop
from composer.core import Event
from composer.core.logging.logger import Logger
from composer.core.state import State
from composer.core.types import DataLoader
from composer.models import MosaicClassifier
from composer.trainer.trainer_hparams import TrainerHparams
from tests.utils.trainer_fit import train_model


@pytest.fixture
def N() -> int:
    """Batch size
    """
    return 16


@pytest.fixture
def D() -> int:
    """Input dimension
    """
    return 8


@pytest.fixture
def X(N: int, D: int) -> torch.Tensor:
    """2D input
    """
    torch.manual_seed(42)
    return torch.randn(N, D)


@pytest.fixture
def X3D(N: int, D: int) -> torch.Tensor:
    """3D input
    """
    torch.manual_seed(42)
    return torch.randn(N, D, D)


@pytest.fixture
def X5D(N: int, D: int) -> torch.Tensor:
    """5D input
    """
    torch.manual_seed(42)
    return torch.randn(N, D, D, D, D)


@pytest.fixture
def Ximage(N: int) -> torch.Tensor:
    """4D image input
    """
    torch.manual_seed(42)
    return torch.randn(N, 3, 32, 32)


@pytest.fixture
def y(N: int) -> torch.Tensor:
    """Target
    """
    torch.manual_seed(42)
    return torch.randint(2, (N,))


@pytest.fixture
def loss_fun() -> Callable:
    """Fake loss function
    """

    def loss(output, target, reduction="none"):
        #import pdb; pdb.set_trace()
        return torch.ones_like(target)

    return loss


@pytest.fixture
def loss_fun_tuple() -> Callable:
    """Fake loss function that requires a batch tuple
    """

    def loss(output, batch, reduction="none"):
        _, target = batch
        return torch.ones_like(target)

    return loss


@pytest.fixture
def bad_loss() -> Callable:
    """Fake loss function that will error
    """

    def loss(output, target):
        return 0

    return loss


@pytest.fixture
def model(X: torch.Tensor) -> torch.nn.Module:
    """Simple fake linear model
    """
    return torch.nn.Linear(X.shape[1], 1)


@pytest.fixture
def model3D(X3D: torch.Tensor) -> torch.nn.Module:
    """Simple fake linear model
    """
    return torch.nn.Sequential(torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(), torch.nn.Linear(X3D.shape[1], 1))


@pytest.fixture
def model5D(X5D: torch.Tensor) -> torch.nn.Module:
    """Simple fake linear model
    """
    return torch.nn.Sequential(torch.nn.AdaptiveAvgPool3d(1), torch.nn.Flatten(), torch.nn.Linear(X5D.shape[1], 1))


@pytest.fixture
def start() -> float:
    """start hparam
    """
    return 0.5


@pytest.fixture
def end() -> float:
    """end hparam
    """
    return 0.8


@pytest.fixture
def keep() -> float:
    """keep hparam
    """
    return 0.5


@pytest.fixture
def scale_factor() -> float:
    """scale_factor hparam
    """
    return 0.5


@pytest.fixture
def interrupt() -> int:
    """interrupt hparams
    """
    return 2


@pytest.fixture
def dummy_hparams(start: float, end: float, keep: float, scale_factor: float,
                  interrupt: int) -> SelectiveBackpropHparams:
    """Dummy algo hparams
    """
    return SelectiveBackpropHparams(start, end, keep, scale_factor, interrupt)


@pytest.fixture
def dummy_algorithm(dummy_hparams: SelectiveBackpropHparams) -> SelectiveBackprop:
    """Dummy algorithm
    """
    return dummy_hparams.initialize_object()


@pytest.fixture
def epoch() -> int:
    """Default epoch
    """
    return 5


@pytest.fixture
def batch() -> int:
    """Default batch
    """
    return 0


@pytest.fixture
def conv_model(Ximage: torch.Tensor, D: int) -> MosaicClassifier:
    """Dummy conv model
    """
    return MosaicClassifier(torch.nn.Conv2d(Ximage.shape[1], D, 3))


@pytest.fixture
def dummy_state_sb(dummy_state: State, dummy_train_dataloader: DataLoader, conv_model: MosaicClassifier,
                   loss_fun_tuple: Callable, epoch: int, batch: int) -> State:
    """Dummy state with required values set for Selective Backprop
    """

    dummy_state.train_dataloader = dummy_train_dataloader
    dummy_state.epoch = epoch
    dummy_state.step = epoch * len(dummy_train_dataloader) + batch
    dummy_state.model = conv_model
    dummy_state.model.module.loss = loss_fun_tuple

    return dummy_state


@pytest.mark.parametrize("epoch,batch,interrupt", [(10, 0, 0), (10, 0, 2), (10, 2, 2)])
def test_do_selective_backprop_true(epoch: int, batch: int, interrupt: int) -> None:
    """Test functional match when epoch is within interval
    """
    start = 5
    end = 15
    is_chosen = do_selective_backprop(epoch, batch, start, end, interrupt)
    assert is_chosen


@pytest.mark.parametrize("epoch,batch,interrupt", [(0, 0, 0), (20, 0, 0), (10, 1, 2)])
def test_do_selective_backprop_false(epoch: int, batch: int, interrupt: int) -> None:
    """Test functional doesn't match when epoch is outside of interval
    """
    start = 5
    end = 15
    is_chosen = do_selective_backprop(epoch, batch, start, end, interrupt)
    assert not is_chosen


@pytest.mark.parametrize("keep", [0.5])
@pytest.mark.parametrize("scale_factor", [0.5])
@pytest.mark.xfail()
def test_selective_output_shape_3D(X3D: torch.Tensor, y: torch.Tensor, model: torch.nn.Module, loss_fun: Callable,
                                   keep: float, scale_factor: float) -> None:
    """Test functional selection on 3D inputs
    """
    N, D, _ = X3D.shape

    X_scaled, y_scaled = selective_backprop(X3D, y, model, loss_fun, keep, scale_factor)
    assert X_scaled.shape == (int(N * keep), D, D)
    assert y_scaled.shape == (int(N * keep),)


@pytest.mark.parametrize("keep", [1, 0.5, 0.75])
@pytest.mark.parametrize("scale_factor", [1])
def test_selective_output_shape(X: torch.Tensor, y: torch.Tensor, model: torch.nn.Module, loss_fun: Callable,
                                keep: float, scale_factor: float) -> None:
    """Test functional selection on 2D inputs
    """
    N, D = X.shape

    X_scaled, y_scaled = selective_backprop(X, y, model, loss_fun, keep, scale_factor)
    assert X_scaled.shape == (int(N * keep), D)
    assert y_scaled.shape == (int(N * keep),)


@pytest.mark.parametrize("keep", [0.5, 0.75, 1])
@pytest.mark.parametrize("scale_factor", [0.5, 0.75])
def test_selective_output_shape_scaled(Ximage: torch.Tensor, y: torch.Tensor, conv_model: MosaicClassifier,
                                       loss_fun: Callable, keep: float, scale_factor: float) -> None:
    """Test functional selection on 4D inputs
    """
    N, C, H, W = Ximage.shape
    X_scaled, y_scaled = selective_backprop(Ximage, y, conv_model.module, loss_fun, keep, scale_factor)
    assert X_scaled.shape == (int(N * keep), C, H, W)
    assert y_scaled.shape == (int(N * keep),)


def test_selective_backprop_interp_dim_error(X: torch.Tensor, y: torch.Tensor, model: torch.nn.Module,
                                             loss_fun: Callable) -> None:
    """Ensure that ValueError is raised when input tensor can't be scaled
    """
    with pytest.raises(ValueError):
        X_scaled, y_scaled = selective_backprop(X, y, model, loss_fun, 1, 0.5)


def test_selective_backprop_bad_loss_error(X: torch.Tensor, y: torch.Tensor, model: torch.nn.Module,
                                           bad_loss: Callable) -> None:
    """Ensure that ValueError is raised when loss function doesn't have `reduction` kwarg
    """
    with pytest.raises(TypeError) as execinfo:
        X_scaled, y_scaled = selective_backprop(X, y, model, bad_loss, 1, 1)
    MATCH = "must take a keyword argument `reduction`."
    assert MATCH in str(execinfo.value)


@pytest.mark.parametrize("event", [Event.AFTER_DATALOADER])
@pytest.mark.parametrize("epoch,batch", [(5, 0), (7, 0), (5, 2)])
def test_match_correct(event: Event, dummy_algorithm: SelectiveBackprop, dummy_state_sb: State) -> None:
    """ Algo should match AFTER_DATALOADER in the right interval
    """
    dummy_state_sb.max_epochs = 10

    assert dummy_algorithm.match(event, dummy_state_sb)


@pytest.mark.parametrize("event,epoch,batch", [(Event.TRAINING_START, 5, 0), (Event.AFTER_DATALOADER, 0, 0),
                                               (Event.AFTER_DATALOADER, 5, 1)])
def test_match_incorrect(event: Event, dummy_algorithm: SelectiveBackprop, dummy_state_sb: State) -> None:
    """ Algo should NOT match TRAINING_START or the wrong interval
    """
    dummy_state_sb.max_epochs = 10

    assert not dummy_algorithm.match(event, dummy_state_sb)


@pytest.mark.parametrize("epoch,batch", [(5, 0)])
@pytest.mark.parametrize("keep", [0.5, 0.75, 1])
@pytest.mark.parametrize("scale_factor", [0.5, 1])
def test_apply(Ximage: torch.Tensor, y: torch.Tensor, dummy_algorithm: SelectiveBackprop, dummy_state_sb: State,
               dummy_logger: Logger, keep: float) -> None:
    """Test apply with image inputs gives the right output shape
    """
    N, C, H, W = Ximage.shape

    dummy_state_sb.max_epochs = 10
    dummy_state_sb.batch = (Ximage, y)
    dummy_algorithm.apply(Event.AFTER_DATALOADER, dummy_state_sb, dummy_logger)

    X_scaled, y_scaled = dummy_state_sb.batch
    assert X_scaled.shape == (int(N * keep), C, H, W)
    assert y_scaled.shape == (int(N * keep),)


@pytest.mark.run_long
@pytest.mark.timeout(90)
def test_selective_backprop_trains(mosaic_trainer_hparams: TrainerHparams):
    mosaic_trainer_hparams.algorithms = [
        SelectiveBackpropHparams(start=0.3, end=0.9, keep=0.75, scale_factor=0.5, interrupt=1)
    ]
    train_model(mosaic_trainer_hparams, max_epochs=6)
