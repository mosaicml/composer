# Copyright 2021 MosaicML. All Rights Reserved.

"""Core Label Smoothing classes and functions."""

from __future__ import annotations

from typing import Optional

import torch

from composer.core.types import Algorithm, Event, Logger, State, Tensor
from composer.models.loss import ensure_targets_one_hot

__all__ = ["LabelSmoothing", "smooth_labels"]


def smooth_labels(logits: Tensor, target: Tensor, smoothing: float = 0.1):
    """Shrink targets towards a uniform distribution as in `Szegedy et al <https://arxiv.org/abs/1512.00567>`_.

    The smoothed labels are computed as ``(1 - smoothing) * targets + smoothing * unif``
    where ``unif`` is a vector with elements all equal to ``1 / num_classes``.

    Args:
        logits (torch.Tensor): predicted value for ``target``, or any other tensor
            with the same shape. Shape must be ``(N, num_classes, ...)`` for
            ``N`` examples and ``num_classes`` classes, with any number of
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
        targets_smooth (torch.Tensor): The smoothed targets

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

    def __init__(self, smoothing: float = 0.1):
        self.smoothing = smoothing
        self.original_labels = torch.Tensor()

    def match(self, event: Event, state: State) -> bool:
        return event in [Event.BEFORE_LOSS, Event.AFTER_LOSS]

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        input, labels = state.batch_pair

        if event == Event.BEFORE_LOSS:
            assert isinstance(state.outputs, Tensor), "Multiple tensors not supported yet"
            assert isinstance(labels, Tensor), "Multiple tensors not supported yet"

            self.original_labels = labels.clone()
            smoothed_labels = smooth_labels(
                state.outputs,
                labels,
                smoothing=self.smoothing,
            )
            state.batch = (input, smoothed_labels)
        elif event == Event.AFTER_LOSS:
            # restore the target to the non-smoothed version
            state.batch = (input, self.original_labels)
