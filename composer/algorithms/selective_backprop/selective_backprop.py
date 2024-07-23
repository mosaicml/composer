# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core SelectiveBackprop class and functions."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import torch
from torch.nn import functional as F

from composer.core import Algorithm, Event, State, get_precision_context
from composer.loggers import Logger
from composer.models import ComposerModel

__all__ = ['SelectiveBackprop', 'select_using_loss', 'should_selective_backprop']


def should_selective_backprop(
    current_duration: float,
    batch_idx: int,
    start: float = 0.5,
    end: float = 0.9,
    interrupt: int = 2,
) -> bool:
    """Decides if selective backprop should be run based on time in training.

    Returns true if the ``current_duration`` is between ``start`` and
    ``end``. It is recommended that SB be applied during the later stages of
    a training run, once the model has already "learned" easy examples.

    To preserve convergence, SB can be interrupted with vanilla minibatch
    gradient steps every ``interrupt`` steps. When ``interrupt=0``, SB will be
    used at every step during the SB interval. When ``interrupt=2``, SB will
    alternate with vanilla minibatch steps.

    Args:
        current_duration (float): The elapsed training duration. Must be
            within ``[0.0, 1.0)``.
        batch_idx (int): The current batch within the epoch.
        start (float, optional): The duration at which selective backprop
            should be enabled, as a percentage. Default: ``0.5``.
        end (float, optional): The duration at which selective backprop
            should be disabled. Default: ``0.9``.
        interrupt (int, optional): The number of batches between vanilla
            minibatch gradient updates. Default: ``2``.

    Returns:
        bool: If selective backprop should be performed on this batch.
    """
    is_interval = ((current_duration >= start) and (current_duration < end))
    is_step = ((interrupt == 0) or ((batch_idx + 1) % interrupt != 0))

    return is_interval and is_step


def select_using_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    model: Callable[[Union[torch.Tensor, Sequence[torch.Tensor]]], torch.Tensor],
    loss_fun: Callable,
    keep: float = 0.5,
    scale_factor: float = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prunes minibatches as a subroutine of :class:`.SelectiveBackprop`. Computes the loss function on the provided training
    examples and runs minibatches according to the difficulty. The fraction of the minibatch that is kept for gradient
    computation is specified by the argument ``0 <= keep <= 1``.

    To speed up SB's selection forward pass, the argument ``scale_factor`` can
    be used to spatially downsample input tensors. The full-sized inputs
    will still be used for the weight gradient computation.

    Args:
        input (torch.Tensor): Input tensor to prune.
        target (torch.Tensor): Output tensor to prune.
        model (Callable): Model with which to predict outputs.
        loss_fun (Callable): Loss function of the form ``loss(outputs, targets, reduction='none')``.
            The function must take the keyword argument ``reduction='none'``
            to ensure that per-sample losses are returned.
        keep (float, optional): Fraction of examples in the batch to keep. Default: ``0.5``.
        scale_factor (float, optional): Multiplier between 0 and 1 for spatial size. Downsampling
            requires the input tensor to be at least 3D. Default: ``1``.

    Returns:
        (torch.Tensor, torch.Tensor): The pruned batch of inputs and targets

    Raises:
        ValueError: If ``scale_factor > 1``.
        TypeError: If ``loss_fun > 1`` has the wrong signature or is not callable.

    .. note::

        This function runs an extra forward pass through the model on the batch of data.
        If you are using a non-default precision, ensure that this forward pass
        runs in your desired precision. For example:

    .. testsetup::

        N_sb, D_sb = 16, 8
        X_sb, y_sb = torch.randn(N_sb, D_sb), torch.randint(2, (N_sb,))
        lin_model = torch.nn.Linear(X_sb.shape[1], 1)

    .. doctest::

        >>> import torch
        >>> from composer.algorithms.selective_backprop import select_using_loss
        >>> with torch.cuda.amp.autocast(True):
        ...     X_new, y_new = select_using_loss(
        ...         X_sb,
        ...         y_sb,
        ...         lin_model,
        ...         loss_fun,
        ...         keep=0.5,
        ...         scale_factor=1
        ...     )
    """
    INTERPOLATE_MODES = {3: 'linear', 4: 'bilinear', 5: 'trilinear'}

    interp_mode = 'bilinear'
    if scale_factor != 1:
        if input.dim() not in INTERPOLATE_MODES:
            raise ValueError(f'Input must be 3D, 4D, or 5D if scale_factor != 1, got {input.dim()}')
        interp_mode = INTERPOLATE_MODES[input.dim()]

    if scale_factor > 1:
        raise ValueError('scale_factor must be <= 1')

    if callable(loss_fun):
        sig = inspect.signature(loss_fun)
        if not 'reduction' in sig.parameters:
            raise TypeError('Loss function `loss_fun` must take a keyword argument `reduction`.')
    else:
        raise TypeError('Loss function must be callable')

    with torch.no_grad():
        N = input.shape[0]

        # Maybe interpolate
        if scale_factor < 1:
            X_scaled = F.interpolate(
                input,
                scale_factor=scale_factor,
                mode=interp_mode,
                align_corners=False,
                recompute_scale_factor=False,
            )
        else:
            X_scaled = input

        # Get per-examples losses
        out = model(X_scaled)
        losses = loss_fun(out, target, reduction='none')

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
    """Selectively backpropagate gradients from a subset of each batch.

    Based on (`Jiang et al, 2019`_), Selective Backprop (SB) prunes minibatches
    according to the difficulty of the individual training examples, and only
    computes weight gradients over the pruned subset, reducing iteration time, and
    speeding up training.

    The fraction of the minibatch that is kept for gradient computation is
    specified by the argument ``0 <= keep <= 1``.

    To speed up SB's selection forward pass, the argument ``scale_factor`` can
    be used to spatially downsample input image tensors. The full-sized inputs
    will still be used for the weight gradient computation.

    To preserve convergence, SB can be interrupted with vanilla minibatch
    gradient steps every ``interrupt`` steps. When ``interrupt=0``, SB will be
    used at every step during the SB interval. When ``interrupt=2``, SB will
    alternate with vanilla minibatch steps.

    .. _Jiang et al, 2019: https://arxiv.org/abs/1910.00762

    Args:
        start (float, optional): SB interval start as fraction of training duration.
            Default: ``0.5``.
        end (float, optional): SB interval end as fraction of training duration.
            Default: ``0.9``.
        keep (float, optional): fraction of minibatch to select and keep for gradient computation.
            Default: ``0.5``.
        scale_factor (float, optional): scale for downsampling input for selection forward pass.
            Default: ``1.``.
        interrupt (int, optional): interrupt SB with a vanilla minibatch step every
            ``interrupt`` batches. Default: ``2``.
        input_key (str | int | tuple[Callable, Callable] | Any, optional): A key that indexes to the input
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair.  The default is 0, which corresponds to any sequence, where the first element
            is the input. Default: ``0``.
        target_key (str | int | tuple[Callable, Callable] | Any, optional): A key that indexes to the target
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair. The default is 1, which corresponds to any sequence, where the second element
            is the target. Default: ``1``.

    Example:
        .. testcode::

            from composer.algorithms import SelectiveBackprop
            algorithm = SelectiveBackprop(start=0.5, end=0.9, keep=0.5)
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[algorithm],
                optimizers=[optimizer]
            )
    """

    def __init__(
        self,
        start: float = 0.5,
        end: float = 0.9,
        keep: float = 0.5,
        scale_factor: float = 1.,
        interrupt: int = 2,
        input_key: Union[str, int, tuple[Callable, Callable], Any] = 0,
        target_key: Union[str, int, tuple[Callable, Callable], Any] = 1,
    ):
        self.start = start
        self.end = end
        self.keep = keep
        self.scale_factor = scale_factor
        self.interrupt = interrupt
        self._loss_fn = None  # set on Event.INIT
        self.input_key, self.target_key = input_key, target_key

    def match(self, event: Event, state: State) -> bool:
        if event == Event.INIT:
            return True
        if event != Event.AFTER_DATALOADER:
            return False

        is_keep = (self.keep < 1)
        if not is_keep:
            return False

        elapsed_duration = state.get_elapsed_duration()
        assert elapsed_duration is not None, 'elapsed duration should be set on Event.AFTER_DATALOADER'

        is_chosen = should_selective_backprop(
            current_duration=float(elapsed_duration),
            batch_idx=int(state.timestamp.batch_in_epoch),
            start=self.start,
            end=self.end,
            interrupt=self.interrupt,
        )
        return is_chosen

    def apply(self, event: Event, state: State, logger: Optional[Logger] = None) -> None:
        if event == Event.INIT:
            if self._loss_fn is None:
                if not isinstance(state.model, ComposerModel):
                    raise RuntimeError('Model must be of type ComposerModel')
                self._loss_fn = state.model.loss
            return

        state.batch = state.device.batch_to_device(state.batch)
        input, target = state.batch_get_item(key=self.input_key), state.batch_get_item(key=self.target_key)
        assert isinstance(input, torch.Tensor) and isinstance(target, torch.Tensor), \
            'Multiple tensors not supported for this method yet.'

        # Model expected to only take in input, not the full batch
        model = lambda X: state.model((X, None))

        def loss(p, y, reduction='none'):
            assert self._loss_fn is not None, 'loss_fn should be set on Event.INIT'
            return self._loss_fn(p, (torch.Tensor(), y), reduction=reduction)

        with get_precision_context(state.precision, state.precision_config):
            new_input, new_target = select_using_loss(input, target, model, loss, self.keep, self.scale_factor)
        state.batch_set_item(self.input_key, new_input)
        state.batch_set_item(self.target_key, new_target)
