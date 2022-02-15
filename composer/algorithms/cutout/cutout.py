# Copyright 2021 MosaicML. All Rights Reserved.
from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch

from composer.core.types import Algorithm, Event, Logger, State, Tensor

log = logging.getLogger(__name__)


def cutout_batch(X: Tensor, n_holes: int = 1, length: Union[int, float] = 0.5) -> Tensor:
    """See :class:`CutOut`.

    Args:
        X (Tensor): Batch Tensor image of size (B, C, H, W).
        n_holes: Integer number of holes to cut out
        length: Side length of the square holes to cut out. Must be greater than
            0. If ``0 < length < 1``, ``length`` is interpreted as a fraction
            of ``min(H, W)`` and converted to ``int(length * min(H, W))``.
            If ``length >= 1``, ``length`` is used as an integer size directly.

    Returns:
        X_cutout: Batch of images with ``n_holes`` holes of dimension
            ``length x length`` replaced with zeros.
    """
    h = X.size(2)
    w = X.size(3)

    if 0 < length < 1:
        length = min(h, w) * length
    length = int(length)

    mask = torch.ones_like(X)
    for _ in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        mask = _generate_mask(mask, w, h, x, y, length)

    X_cutout = X * mask
    return X_cutout


class CutOut(Algorithm):
    """`Cutout <https://arxiv.org/abs/1708.04552>`_ is a data augmentation technique that works by masking out one or
    more square regions of an input image.

    This implementation cuts out the same square from all images in a batch.

    Args:
        X (Tensor): Batch Tensor image of size (B, C, H, W).
        n_holes: Integer number of holes to cut out
        length: Side length of the square holes to cut out. Must be greater than
            0. If ``0 < length < 1``, ``length`` is interpreted as a fraction
            of ``min(H, W)`` and converted to ``int(length * min(H, W))``.
            If ``length >= 1``, ``length`` is used as an integer size directly.
    """

    def __init__(self, n_holes: int = 1, length: Union[int, float] = 0.5):
        self.n_holes = n_holes
        self.length = length

    def match(self, event: Event, state: State) -> bool:
        """Runs on Event.AFTER_DATALOADER."""
        return event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Apply cutout on input images."""
        x, y = state.batch_pair
        assert isinstance(x, Tensor), "Multiple tensors not supported for Cutout."

        new_x = cutout_batch(X=x, n_holes=self.n_holes, length=self.length)
        state.batch = (new_x, y)


def _generate_mask(mask: Tensor, width: int, height: int, x: int, y: int, cutout_length: int) -> Tensor:
    y1 = np.clip(y - cutout_length // 2, 0, height)
    y2 = np.clip(y + cutout_length // 2, 0, height)
    x1 = np.clip(x - cutout_length // 2, 0, width)
    x2 = np.clip(x + cutout_length // 2, 0, width)

    mask[:, :, y1:y2, x1:x2] = 0.

    return mask
