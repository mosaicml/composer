# Copyright 2021 MosaicML. All Rights Reserved.

import logging
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import yahp as hp

from composer.algorithms import AlgorithmHparams
from composer.core.types import Algorithm, Event, Logger, State, Tensor

log = logging.getLogger(__name__)


def generate_mask(mask: Tensor, width: int, height: int, x: int, y: int, cutout_length: int) -> Tensor:
    y1 = np.clip(y - cutout_length // 2, 0, height)
    y2 = np.clip(y + cutout_length // 2, 0, height)
    x1 = np.clip(x - cutout_length // 2, 0, width)
    x2 = np.clip(x + cutout_length // 2, 0, width)

    mask[:, :, y1:y2, x1:x2] = 0.

    return mask


def apply_cutout(X: Tensor, mask: Tensor):
    return X * mask


def cutout(X: Tensor, n_holes: int, length: int) -> Tensor:
    """See :class:`CutOut`.

    Args:
        X (Tensor): Batch Tensor image of size (B, C, H, W).
        n_holes: Integer number of holes to cut out
        length: Side length of the square hole to cut out.

    Returns:
        X_cutout: Image with `n_holes` of dimension `length x length` cut out of it.
    """
    h = X.size(2)
    w = X.size(3)

    mask = torch.ones_like(X)
    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        mask = generate_mask(mask, w, h, x, y, length)

    X_cutout = apply_cutout(X, mask)
    return X_cutout


@dataclass
class CutOutHparams(AlgorithmHparams):
    """See :class:`CutOut`"""

    n_holes: int = hp.required('Number of holes to cut out', template_default=1)
    length: int = hp.required('Side length of the square hole to cut out', template_default=112)

    def initialize_object(self) -> "CutOut":
        return CutOut(**asdict(self))


class CutOut(Algorithm):
    """`Cutout <https://arxiv.org/abs/1708.04552>`_ is a data augmentation
    technique that works by masking out one or more square regions of an
    input image.

    This implementation cuts out the same square from all images in a batch.

    Args:
        X (Tensor): Batch Tensor image of size (B, C, H, W).
        n_holes: Integer number of holes to cut out
        length: Side length of the square hole to cut out.
    """

    def __init__(self, n_holes: int, length: int):
        self.hparams = CutOutHparams(n_holes=n_holes, length=length)

    def match(self, event: Event, state: State) -> bool:
        """Runs on Event.AFTER_DATALOADER"""
        return event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Apply cutout on input images"""
        x, y = state.batch_pair
        assert isinstance(x, Tensor), "Multiple tensors not supported for Cutout."

        new_x = cutout(X=x, **asdict(self.hparams))
        state.batch = (new_x, y)
