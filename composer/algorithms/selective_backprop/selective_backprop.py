# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import inspect
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from composer.core.types import Algorithm, Event, Logger, State, Tensor, Tensors
from composer.models import ComposerModel


def should_selective_backprop(
    current_duration: float,
    batch_idx: int,
    start: float = 0.5,
    end: float = 0.9,
    interrupt: int = 2,
) -> bool:
    """Decide if selective backprop should be run based on time in training.

    Returns true if the ``current_duration`` is between ``start`` and
    ``end``. Recommend that SB be applied during the later stages of
    a training run, once the model has already "learned" easy examples.

    To preserve convergence, SB can be interrupted with vanilla minibatch
    gradient steps every ``interrupt`` steps. When ``interrupt=0``, SB will be
    used at every step during the SB interval. When ``interrupt=2``, SB will
    alternate with vanilla minibatch steps.

    Args:
        current_duration (float): The elapsed training duration. Must be
            within :math:`[0.0, 1.0)`.
        batch_idx (int): The current batch within the epoch
        start (float, optional): The duration at which selective backprop
            should be enabled. Default: ``0.5``.
        end (float, optional): The duration at which selective backprop
            should be disabled Default: ``0.9``.
        interrupt (int, optional): The number of batches between vanilla
            minibatch gradient updates Default: ``2``.

    Returns:
        bool: If selective backprop should be performed on this batch.
    """
    is_interval = ((current_duration >= start) and (current_duration < end))
    is_step = ((interrupt == 0) or ((batch_idx + 1) % interrupt != 0))

    return is_interval and is_step


def select_using_loss(input: torch.Tensor,
                      target: torch.Tensor,
                      model: Callable[[Tensors], Tensor],
                      loss_fun: Callable,
                      keep: float = 0.5,
                      scale_factor: float = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Selectively backpropagate gradients from a subset of each batch (`Jiang et al, 2019 <https://\\
    arxiv.org/abs/1910.00762>`_).

    Selective Backprop (SB) prunes minibatches according to the difficulty
    of the individual training examples and only computes weight gradients
    over the selected subset. This reduces iteration time and speeds up training.
    The fraction of the minibatch that is kept for gradient computation is
    specified by the argument ``0 <= keep <= 1``.

    To speed up SB's selection forward pass, the argument ``scale_factor`` can
    be used to spatially downsample input tensors. The full-sized inputs
    will still be used for the weight gradient computation.

    Args:
        input (torch.Tensor): Input tensor to prune
        target (torch.Tensor): Output tensor to prune
        model (Callable): Model with which to predict outputs
        loss_fun (Callable): Loss function of the form ``loss(outputs, targets, reduction='none')``.
            The function must take the keyword argument ``reduction='none'``
            to ensure that per-sample losses are returned.
        keep (float, optional): Fraction of examples in the batch to keep. Default: ``0.5``.
        scale_factor (float, optional): Multiplier between 0 and 1 for spatial size. Downsampling
            requires the input tensor to be at least 3D. Default: ``1``.

    Returns:
        (torch.Tensor, torch.Tensor): The pruned batch of inputs and targets

    Raises:
        ValueError: If ``scale_factor > 1``
        TypeError: If ``loss_fun > 1`` has the wrong signature or is not callable

    Note:
    This function runs an extra forward pass through the model on the batch of data.
    If you are using a non-default precision, ensure that this forward pass
    runs in your desired precision. For example:

    .. code-block:: python

        with torch.cuda.amp.autocast(True):
            X_new, y_new = selective_backprop(X, y, model, loss_fun, keep, scale_factor)
    """
    INTERPOLATE_MODES = {3: "linear", 4: "bilinear", 5: "trilinear"}

    interp_mode = "bilinear"
    if scale_factor != 1:
        if input.dim() not in INTERPOLATE_MODES:
            raise ValueError(f"Input must be 3D, 4D, or 5D if scale_factor != 1, got {input.dim()}")
        interp_mode = INTERPOLATE_MODES[input.dim()]

    if scale_factor > 1:
        raise ValueError("scale_factor must be <= 1")

    if callable(loss_fun):
        sig = inspect.signature(loss_fun)
        if not "reduction" in sig.parameters:
            raise TypeError("Loss function `loss_fun` must take a keyword argument `reduction`.")
    else:
        raise TypeError("Loss function must be callable")

    with torch.no_grad():
        N = input.shape[0]

        # Maybe interpolate
        if scale_factor < 1:
            X_scaled = F.interpolate(input, scale_factor=scale_factor, mode=interp_mode)
        else:
            X_scaled = input

        # Get per-examples losses
        out = model(X_scaled)
        losses = loss_fun(out, target, reduction="none")

        # Sort losses
        sorted_idx = torch.argsort(losses)
        n_select = int(keep * N)

        # Sample by loss
        percs = np.arange(0.5, N, 1) / N
        probs = percs**((1.0 / keep) - 1.0)
        probs = probs / np.sum(probs)
        select_percs_idx = np.random.choice(N, n_select, replace=False, p=probs)
        select_idx = sorted_idx[select_percs_idx]

    return input[select_idx], target[select_idx]


class SelectiveBackprop(Algorithm):
    """Selectively backpropagate gradients from a subset of each batch (`Jiang et al, 2019 <https://\\
    arxiv.org/abs/1910.00762>`_).

    Selective Backprop (SB) prunes minibatches according to the difficulty
    of the individual training examples, and only computes weight gradients
    over the pruned subset, reducing iteration time and speeding up training.
    The fraction of the minibatch that is kept for gradient computation is
    specified by the argument ``0 <= keep <= 1``.

    To speed up SB's selection forward pass, the argument ``scale_factor`` can
    be used to spatially downsample input image tensors. The full-sized inputs
    will still be used for the weight gradient computation.

    To preserve convergence, SB can be interrupted with vanilla minibatch
    gradient steps every ``interrupt`` steps. When ``interrupt=0``, SB will be
    used at every step during the SB interval. When ``interrupt=2``, SB will
    alternate with vanilla minibatch steps.

    Args:
        start (float, optional): SB interval start as fraction of training duration
            Default: ``0.5``.
        end (float, optional): SB interval end as fraction of training duration
            Default: ``0.9``.
        keep (float, optional): fraction of minibatch to select and keep for gradient computation
            Default: ``0.5``.
        scale_factor (float, optional): scale for downsampling input for selection forward pass
            Default: ``0.5``.
        interrupt (int, optional): interrupt SB with a vanilla minibatch step every
            ``interrupt`` batches. Default: ``2``.
    """

    def __init__(self,
                 start: float = 0.5,
                 end: float = 0.9,
                 keep: float = 0.5,
                 scale_factor: float = 0.5,
                 interrupt: int = 2):
        self.start = start
        self.end = end
        self.keep = keep
        self.scale_factor = scale_factor
        self.interrupt = interrupt
        self._loss_fn = None  # set on Event.INIT

    def match(self, event: Event, state: State) -> bool:
        """Matches :attr:`Event.INIT` and `Event.AFTER_DATALOADER`

        * Uses `Event.INIT` to get the loss function before the model is wrapped
        * Uses `Event.AFTER_DATALOADER`` to apply selective backprop if time is between ``self.start`` and ``self.end``.
        """
        if event == Event.INIT:
            return True
        if event != Event.AFTER_DATALOADER:
            return False

        is_keep = (self.keep < 1)
        if not is_keep:
            return False

        is_chosen = should_selective_backprop(
            current_duration=float(state.get_elapsed_duration()),
            batch_idx=state.timer.batch_in_epoch.value,
            start=self.start,
            end=self.end,
            interrupt=self.interrupt,
        )
        return is_chosen

    def apply(self, event: Event, state: State, logger: Optional[Logger] = None) -> None:
        """Apply selective backprop to the current batch."""
        if event == Event.INIT:
            if self._loss_fn is None:
                if not isinstance(state.model, ComposerModel):
                    raise RuntimeError("Model must be of type ComposerModel")
                self._loss_fn = state.model.loss
            return
        input, target = state.batch_pair
        assert isinstance(input, Tensor) and isinstance(target, Tensor), \
            "Multiple tensors not supported for this method yet."

        # Model expected to only take in input, not the full batch
        model = lambda X: state.model((X, None))

        def loss(p, y, reduction="none"):
            assert self._loss_fn is not None, "loss_fn should be set on Event.INIT"
            return self._loss_fn(p, (torch.Tensor(), y), reduction=reduction)

        with state.precision_context:
            new_input, new_target = select_using_loss(input, target, model, loss, self.keep, self.scale_factor)
        state.batch = (new_input, new_target)
