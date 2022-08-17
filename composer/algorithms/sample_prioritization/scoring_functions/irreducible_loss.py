# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict

from torch import Tensor

from composer.algorithms.sample_prioritization import register_scoring_fxn


@register_scoring_fxn('irreducible_loss')  #  type: ignore
def irreducible_loss(logits: Tensor, targets: Tensor, loss_fxn: Callable, **kwargs: Dict):
    """'Irreducible_loss' as described in (`Mindermann et al. 2021
    <https://arxiv.org/abs/2107.02565>`_). Essentially surrogate_loss - current_loss."""
    if 'score' not in kwargs.keys():
        raise KeyError(
            "The irreducible loss scoring function requires a dataset that returns samples as a dict with the key 'score'."
        )
    return kwargs['score'] - loss_fxn(logits, targets)
