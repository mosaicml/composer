# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core Sample Prioritization class and functions."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from maskedtensor import masked_tensor
from torch.nn.functional import cross_entropy

from composer.algorithms.selective_backprop import should_selective_backprop
from composer.core import Algorithm, Event, State
from composer.core.precision import get_precision_context
from composer.loggers import Logger
from composer.models import ComposerModel

__all__ = ['SamplePrioritization', 'select_using_loss']


def masked_median(mskd_tensor, dim=1) -> torch.Tensor:
    values = mskd_tensor.to_tensor(torch.nan)
    return torch.nanmedian(values, dim=dim).values


def masked_l2(mskd_tensor, dim=1) -> torch.Tensor:
    values = mskd_tensor.to_tensor(0)
    return torch.norm(values, p=2, dim=dim)


def masked_mean(mskd_tensor, dim=1) -> torch.Tensor:
    return torch.mean(mskd_tensor, dim=dim).to_tensor(0)


def masked_max(mskd_tensor, dim=1) -> torch.Tensor:
    return torch.amax(mskd_tensor, dim=dim).to_tensor(0)


def masked_min(mskd_tensor, dim=1) -> torch.Tensor:
    return torch.amin(mskd_tensor, dim=dim).to_tensor(0)


def rank_and_select(
    losses,
    selection_metric: str = 'mean',
    pct_keep: float = 0.5,
    keep_from: str = 'bottom',
    sample: bool = False,
):

    metrics = {
        'max': masked_max,
        'min': masked_min,
        'mean': masked_mean,  #this is selective backprop as modestly adapted
        'median': masked_median,
        'l2': masked_l2,
    }

    loss_ranks = metrics[selection_metric](losses)

    # Sort losses
    sorted_idx = torch.argsort(loss_ranks)
    N = loss_ranks.size()[0]
    n_select = int(pct_keep * N)
    if sample:
        # Sample by loss
        percs = np.arange(0.5, N, 1) / N
        probs = percs**((1.0 / pct_keep) - 1.0)
        probs = probs / np.sum(probs)
        select_percs_idx = np.random.choice(N, n_select, replace=False, p=probs)
        select_idx = sorted_idx[select_percs_idx]
    else:
        select_idx = keep_logic(keep_from, sorted_idx, n_select)
    return select_idx


def keep_logic(keep_from, sorted_idx, n_select):
    if keep_from == 'bottom':
        select_idx = sorted_idx < n_select
    elif keep_from == 'top':
        select_idx = sorted_idx >= n_select
    elif keep_from == 'middle':
        select_idx = torch.logical_or(sorted_idx >= len(sorted_idx) - (n_select // 2), sorted_idx < (n_select // 2))
    else:
        select_idx = sorted_idx >= 0
    return select_idx


def select_using_loss(input,
                      target: torch.Tensor,
                      model: Callable[[Union[torch.Tensor, Sequence[torch.Tensor]]], torch.Tensor],
                      loss_fun: Callable = cross_entropy,
                      pct_keep: float = 0.5,
                      ignore_index: int = -100,
                      selection_metric: str = 'mean',
                      keep_from: str = 'bottom') -> Tuple[torch.Tensor, torch.Tensor]:
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
        pct_keep (float, optional): Fraction of examples in the batch to keep. Default: ``0.5``.

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

    if not callable(loss_fun):
        raise TypeError('Loss function must be callable')

    with torch.no_grad():

        # Get per-examples losses
        output = model(input)
        losses = loss_fun(output['logits'].view(-1, model.config.vocab_size), target.view(-1),
                          reduction='none').view_as(target)
        losses = masked_tensor(losses, target != ignore_index)
        select_idx = rank_and_select(losses, selection_metric, pct_keep, keep_from)
        target = target[select_idx]
        for input_key in input.keys():
            input[input_key] = input[input_key][select_idx]

    return input, target


class SamplePrioritization(Algorithm):
    """Selectively backpropagate gradients from a subset of each batch.

    Inspired by on Selective Backprop (SB) (`Jiang et al, 2019`_). Prunes minibatches
    according to the difficulty of the individual training examples, and only
    computes weight gradients over the pruned subset, reducing iteration time, and
    speeding up training.

    The fraction of the minibatch that is kept for gradient computation is
    specified by the argument ``0 <= keep <= 1``.

    To preserve convergence, it can be interrupted with vanilla minibatch
    gradient steps every ``interrupt`` steps. When ``interrupt=0``, SB will be
    used at every step during the SB interval. When ``interrupt=2``, SB will
    alternate with vanilla minibatch steps.

    .. _Jiang et al, 2019: https://arxiv.org/abs/1910.00762

    Args:
        start (float, optional): SB interval start as fraction of training duration.
            Default: ``0.0``.
        end (float, optional): SB interval end as fraction of training duration.
            Default: ``1.0``.
        pct_keep (float, optional): fraction of minibatch to select and keep for gradient computation.
            Default: ``0.5``.
        interrupt (int, optional): interrupt SB with a vanilla minibatch step every
            ``interrupt`` batches. Default: ``0``.
        selection_metric (str, optional): how to select using loss. Default: ``mean``.
        keep_from (str, optional): where in the distribution to remove samples. Default: ``bottom``.
        target_key (str | int | Tuple[Callable, Callable] | Any, optional): A key that indexes to the target
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair. The default is 1, which corresponds to any sequence, where the second element
            is the target. Default: ``labels``.
        loss_fn (Callable[[torch.Tensor], torch.Tensor], optional): the loss function to use to calculate the heuristic.  Default: ``cross_entropy``.
        ignore_index (int, optional): the integer used to denote the mask token. Default ``-100``.

    Example:
        .. testcode::

            from composer.algorithms import SamplePrioritzation
            algorithm = SelectiveBackprop(start=0.0, end=1.0, keep=0.5)
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
        start: float = 0.0,
        end: float = 1.0,
        pct_keep: float = 0.5,
        interrupt: int = 0,
        selection_metric: str = 'mean',
        keep_from: str = 'bottom',
        target_key: str = 'labels',
        loss_fn: Callable[[torch.Tensor], torch.Tensor] = cross_entropy,
        ignore_index: int = -100,
    ):
        self.start = start
        self.end = end
        self.pct_keep = pct_keep
        self.interrupt = interrupt
        self._loss_fn = loss_fn
        self.target_key = target_key
        self.selection_metric = selection_metric
        self.keep_from = keep_from
        self.ignore_index = ignore_index

    def match(self, event: Event, state: State) -> bool:
        if event == Event.INIT:
            return True
        if event != Event.AFTER_DATALOADER:
            return False

        is_keep = (self.pct_keep < 1)
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
        target = state.batch.pop(self.target_key)

        with get_precision_context(state.precision):
            new_input, new_target = select_using_loss(state.batch, target, state.model, self._loss_fn, self.pct_keep,
                                                      self.ignore_index, self.selection_metric, self.keep_from)

        for key in new_input.keys():
            state.batch_set_item(key, new_input[key])
        state.batch['labels'] = new_target
