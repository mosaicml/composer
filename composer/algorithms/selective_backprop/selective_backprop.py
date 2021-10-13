# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import yahp as hp
from torch.nn import functional as F

from composer.algorithms.algorithm_hparams import AlgorithmHparams
from composer.core.types import Algorithm, Event, Logger, State, Tensor


def do_selective_backprop(
    epoch: int,
    batch_idx: int,
    start_epoch: int,
    end_epoch: int,
    interrupt: int,
) -> bool:
    """Decide if selective backprop should be run based on time in training.

    Returns true if the current ``epoch`` is between ``start_epoch`` and
    ``end_epoch``. Recommend that SB be applied during the later stages of
    a training run, once the model has already "learned" easy examples.

    To preserve convergence, SB can be interrupted with vanilla minibatch
    gradient steps every ``interrupt`` steps. When ``interrupt=0``, SB will be
    used at every step during the SB interval. When ``interrupt=2``, SB will
    alternate with vanilla minibatch steps.

    Args:
        epoch: The current epoch during training
        batch_idx: The current batch within the epoch
        start_epoch: The epoch at which selective backprop should be enabled
        end_epoch: The epoch at which selective backprop should be disabled
        interrupt: The number of batches between vanilla minibatch gradient updates

    Returns:
        bool: If selective backprop should be performed on this batch.
    """
    is_interval = ((epoch >= start_epoch) and (epoch < end_epoch))
    is_step = ((interrupt == 0) or ((batch_idx + 1) % interrupt != 0))

    return is_interval and is_step


# TODO this function should probably be part of the public API
def selective_backprop(X: torch.Tensor,
                       y: torch.Tensor,
                       model: torch.nn.Module,
                       loss_fun: Callable,
                       keep: float,
                       scale_factor: float = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select a subset of the batch on which to learn as per (`Jiang et al. 2019 <https://arxiv.org/abs/1910.00762>`_)

    Selective Backprop (SB) prunes minibatches according to the difficulty
    of the individual training examples and only computes weight gradients
    over the selected subset. This reduces iteration time and speeds up training.
    The fraction of the minibatch that is kept for gradient computation is
    specified by the argument ``0 <= keep <= 1``.

    To speed up SB's selection forward pass, the argument ``scale_factor`` can
    be used to spatially downsample input tensors. The full-sized inputs
    will still be used for the weight gradient computation.

    Args:
        X: Input tensor to prune
        y: Output tensor to prune
        model: Model with which to predict outputs
        loss_fun: Loss function of the form ``loss(outputs, targets, reduction='none')``.
            The function must take the keyword argument ``reduction='none'``
            to ensure that per-sample losses are returned.
        keep: Fraction of examples in the batch to keep
        scale_factor: Multiplier between 0 and 1 for spatial size. Downsampling
            requires the input tensor to be at least 3D.

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
        if X.dim() not in INTERPOLATE_MODES:
            raise ValueError(f"Input must be 3D, 4D, or 5D if scale_factor != 1, got {X.dim()}")
        interp_mode = INTERPOLATE_MODES[X.dim()]

    if scale_factor > 1:
        raise ValueError("scale_factor must be <= 1")

    if callable(loss_fun):
        sig = inspect.signature(loss_fun)
        if not "reduction" in sig.parameters:
            raise TypeError("Loss function `loss_fun` must take a keyword argument `reduction`.")
    else:
        raise TypeError("Loss function must be callable")

    with torch.no_grad():
        N = X.shape[0]

        # Maybe interpolate
        if scale_factor < 1:
            X_scaled = F.interpolate(X, scale_factor=scale_factor, mode=interp_mode)
        else:
            X_scaled = X

        # Get per-examples losses
        out = model(X_scaled)
        losses = loss_fun(out, y, reduction="none")

        # Sort losses
        sorted_idx = torch.argsort(losses)
        n_select = int(keep * N)

        # Sample by loss
        percs = np.arange(0.5, N, 1) / N
        probs = percs**((1.0 / keep) - 1.0)
        probs = probs / np.sum(probs)
        select_percs_idx = np.random.choice(N, n_select, replace=False, p=probs)
        select_idx = sorted_idx[select_percs_idx]

    return X[select_idx], y[select_idx]


@dataclass
class SelectiveBackpropHparams(AlgorithmHparams):
    """See :class:`SelectiveBackprop`"""

    start: float = hp.required(doc="SB interval start, as fraction of training duration", template_default=0.5)
    end: float = hp.required(doc="SB interval end, as fraction of training duration", template_default=0.9)
    keep: float = hp.required(doc="fraction of minibatch to select and keep for gradient computation",
                              template_default=0.5)
    scale_factor: float = hp.required(doc="scale for downsampling input for selection forward pass",
                                      template_default=0.5)
    interrupt: int = hp.required(doc="interrupt SB with a vanilla minibatch step every 'interrupt' batches",
                                 template_default=2)

    def initialize_object(self) -> SelectiveBackprop:
        return SelectiveBackprop(**asdict(self))


class SelectiveBackprop(Algorithm):
    """Selectively backpropagate gradients from a subset of each batch (`Jiang et al. 2019 <https://arxiv.org/abs/1910.00762>`_).

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
        start: SB interval start as fraction of training duration
        end: SB interval end as fraction of training duration
        keep: fraction of minibatch to select and keep for gradient computation
        scale_factor: scale for downsampling input for selection forward pass
        interrupt: interrupt SB with a vanilla minibatch step every
            ``interrupt`` batches
    """

    def __init__(self, start: float, end: float, keep: float, scale_factor: float, interrupt: int):
        self.hparams = SelectiveBackpropHparams(start=start,
                                                end=end,
                                                keep=keep,
                                                scale_factor=scale_factor,
                                                interrupt=interrupt)

    def match(self, event: Event, state: State) -> bool:
        """Match on ``Event.AFTER_DATALOADER`` if time is between ``self.start`` and
        ``self.end``."""
        is_event = (event == Event.AFTER_DATALOADER)
        if not is_event:
            return False

        is_keep = (self.hparams.keep < 1)
        if not is_keep:
            return False

        is_chosen = do_selective_backprop(
            epoch=state.epoch,
            batch_idx=state.batch_idx,
            start_epoch=int(state.max_epochs * self.hparams.start),
            end_epoch=int(state.max_epochs * self.hparams.end),
            interrupt=self.hparams.interrupt,
        )
        return is_chosen

    def apply(self, event: Event, state: State, logger: Optional[Logger] = None) -> None:
        """Apply selective backprop to the current batch."""
        input, target = state.batch_pair
        assert isinstance(input, Tensor) and isinstance(target, Tensor), \
            "Multiple tensors not supported for this method yet."

        assert callable(state.model.module.loss)  # type: ignore - type not found

        # Model expected to only take in input, not the full batch
        model = lambda X: state.model((X, None))

        def loss(p, y, reduction="none"):
            return state.model.module.loss(p, (None, y), reduction=reduction)  # type: ignore

        with state.precision_context(state.precision):
            new_input, new_target = selective_backprop(
                input, target, model, loss, self.hparams.keep,
                self.hparams.scale_factor)  # type: ignore - ditto because of loss
        state.batch = (new_input, new_target)
