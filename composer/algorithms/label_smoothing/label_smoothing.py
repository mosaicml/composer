# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass
from typing import Optional

import yahp as hp

from composer.algorithms.algorithm_hparams import AlgorithmHparams
from composer.core.types import Algorithm, Event, Logger, State, Tensor
from composer.models.loss import ensure_targets_one_hot


@dataclass
class LabelSmoothingHparams(AlgorithmHparams):

    alpha: float = hp.required(doc='smoothing factor', template_default=0.1)

    def initialize_object(self) -> "LabelSmoothing":
        return LabelSmoothing(**asdict(self))


class LabelSmoothing(Algorithm):
    """Applies label smoothing during before_loss, then restores the
       original labels during after_loss.

       Args:
        alpha (float): Strength of the label smoothing, between [0, 1].
            alpha=0 means no label smoothing, and alpha=1 means maximal
            smoothing (targets are ignored)
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
    """Shrinks targets towards a prior distribution to counteract label noise.

    This is computed by `(1 - alpha) * targets + alpha * smoothed_targets`
    where `smoothed_targets` is a pre-specified vector of class probabilities.

    Introduced in: https://arxiv.org/abs/1512.00567
    Evaluated in: https://arxiv.org/abs/1906.02629

    Args:
        logits: Output of the model. Tensor of shape (N, C, d1, ..., dn) for
            N examples and C classes, and d1, ..., dn extra dimensions.
        targets: Tensor of shape (N) containing integers 0 <= i <= C-1
            specifying the target labels for each example.
        alpha: Strength of the label smoothing, between [0, 1].
            alpha=0 means no label smoothing, and alpha=1 means maximal
            smoothing (targets are ignored)
    """

    targets = ensure_targets_one_hot(logits, targets)
    n_classes = logits.shape[1]
    return (targets * (1. - alpha)) + (alpha / n_classes)
