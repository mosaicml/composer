# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass
from typing import Optional

import yahp as hp

from composer.algorithms.algorithm_hparams import AlgorithmHparams
from composer.core.types import Algorithm, Event, Logger, State, Tensor
from composer.models.loss import ensure_targets_one_hot


@dataclass
class LabelSmoothingHparams(AlgorithmHparams):
    """See :class:`LabelSmoothing`"""

    alpha: float = hp.required(doc='smoothing factor', template_default=0.1)

    def initialize_object(self) -> "LabelSmoothing":
        return LabelSmoothing(**asdict(self))


class LabelSmoothing(Algorithm):
    """Shrinks targets towards a uniform distribution to counteract label noise
    as in `Szegedy et al. <https://arxiv.org/abs/1512.00567>`_.

    This is computed by ``(1 - alpha) * targets + alpha * smoothed_targets``
    where ``smoothed_targets`` is a vector of ones.

    Introduced in `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`_.

    Args:
        alpha: Strength of the label smoothing, in [0, 1]. ``alpha=0``
            means no label smoothing, and ``alpha=1`` means maximal
            smoothing (targets are ignored).
    """

    def __init__(self, alpha: float):
        self.hparams = LabelSmoothingHparams(alpha=alpha)

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
                alpha=self.hparams.alpha,
            )
            state.batch = (input, smoothed_labels)
        elif event == Event.AFTER_LOSS:
            # restore the target to the non-smoothed version
            state.batch = (input, self.original_labels)


def smooth_labels(logits: Tensor, targets: Tensor, alpha: float):
    """Shrinks targets towards a uniform distribution to counteract label noise
    as in `Szegedy et al. <https://arxiv.org/abs/1512.00567>`_.

    This is computed by ``(1 - alpha) * targets + alpha * smoothed_targets``
    where ``smoothed_targets`` is a vector of ones.

    Args:
        logits: Output of the model. Tensor of shape (N, C, d1, ..., dn) for
            N examples and C classes, and d1, ..., dn extra dimensions.
        targets: Tensor of shape (N) containing integers 0 <= i <= C-1
            specifying the target labels for each example.
        alpha: Strength of the label smoothing, in [0, 1]. ``alpha=0``
            means no label smoothing, and ``alpha=1`` means maximal
            smoothing (targets are ignored).
    """

    targets = ensure_targets_one_hot(logits, targets)
    n_classes = logits.shape[1]
    return (targets * (1. - alpha)) + (alpha / n_classes)
