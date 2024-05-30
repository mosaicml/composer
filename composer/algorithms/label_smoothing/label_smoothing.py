# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core Label Smoothing classes and functions."""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.loss.utils import ensure_targets_one_hot

__all__ = ['LabelSmoothing', 'smooth_labels']


def smooth_labels(logits: torch.Tensor, target: torch.Tensor, smoothing: float = 0.1):
    """Shrink targets towards a uniform distribution as in `Szegedy et al <https://arxiv.org/abs/1512.00567>`_.

    The smoothed labels are computed as ``(1 - smoothing) * targets + smoothing * unif``
    where ``unif`` is a vector with elements all equal to ``1 / num_classes``.

    Args:
        logits (torch.Tensor): predicted value for ``target``, or any other tensor
            with the same shape. Shape must be ``(N, num_classes, ...)`` for
            ``N`` examples and ``num_classes`` classes with any number of
            optional extra dimensions.
        target (torch.Tensor): target tensor of either shape ``N`` or
            ``(N, num_classes, ...)``. In the former case, elements of
            ``targets`` must be integer class ids in the range
            ``0..num_classes``. In the latter case, ``targets`` must have the
            same shape as ``logits``.
        smoothing (float, optional): strength of the label smoothing, in
            :math:`[0, 1]`. ``smoothing=0`` means no label smoothing, and
            ``smoothing=1`` means maximal smoothing (targets are ignored).
            Default: ``0.1``.

    Returns:
        torch.Tensor: The smoothed targets.

    Example:
        .. testcode::

            import torch

            num_classes = 10
            targets = torch.randint(num_classes, size=(100,))
            from composer.algorithms.label_smoothing import smooth_labels
            new_targets = smooth_labels(logits=logits,
                                        target=targets,
                                        smoothing=0.1)
    """

    target = ensure_targets_one_hot(logits, target)
    n_classes = logits.shape[1]
    return (target * (1. - smoothing)) + (smoothing / n_classes)


class LabelSmoothing(Algorithm):
    """Shrink targets towards a uniform distribution as in `Szegedy et al <https://arxiv.org/abs/1512.00567>`_.

    The smoothed labels are computed as ``(1 - smoothing) * targets + smoothing * unif``
    where ``unif`` is a vector with elements all equal to ``1 / num_classes``.

    Args:
        smoothing: Strength of the label smoothing, in :math:`[0, 1]`.
            ``smoothing=0`` means no label smoothing, and
            ``smoothing=1`` means maximal smoothing (targets are ignored).
            Default: ``0.1``.
        target_key (str | int | tuple[Callable, Callable] | Any, optional): A key that indexes to the target
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair. The default is 1, which corresponds to any sequence, where the second element
            is the target. Default: ``1``.

    Example:
        .. testcode::

            from composer.algorithms import LabelSmoothing
            algorithm = LabelSmoothing(smoothing=0.1)
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
        smoothing: float = 0.1,
        target_key: Union[str, int, tuple[Callable, Callable], Any] = 1,
    ):
        self.smoothing = smoothing
        self.original_labels = torch.Tensor()
        self.target_key = target_key

    def match(self, event: Event, state: State) -> bool:
        return event in [Event.BEFORE_LOSS, Event.AFTER_LOSS]

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        labels = state.batch_get_item(self.target_key)

        if event == Event.BEFORE_LOSS:
            assert isinstance(state.outputs, torch.Tensor), 'Multiple tensors not supported yet'
            assert isinstance(labels, torch.Tensor), 'Multiple tensors not supported yet'

            self.original_labels = labels.clone()
            smoothed_labels = smooth_labels(
                state.outputs,
                labels,
                smoothing=self.smoothing,
            )
            state.batch_set_item(self.target_key, smoothed_labels)
        elif event == Event.AFTER_LOSS:
            # restore the target to the non-smoothed version
            state.batch_set_item(self.target_key, self.original_labels)
