# Copyright 2022 MosaicML. All Rights Reserved.

import logging
from functools import reduce
from operator import mul
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from composer.core import Algorithm, Event, State
from composer.loggers import Logger

log = logging.getLogger(__name__)


def dropblock_batch(x: Tensor, drop_prob: float = 0.1, block_size: int = 7, batchwise: bool = True):
    """See :class:`DropBlock`.

    Args:
        x (Tensor): Original data.
        drop_prob (float, optional): Drop probability. Default: 0.1.
        block_size (int, optional): Size of blocks to drop out. Default: 7.
        batchwise (bool, optional): Whether to mask per batch (faster) or per sample. Default: True.

    Returns:
        y (Tensor): Data with dropblock applied.

    Example:
        .. testcode::

           from composer.algorithms.dropblock import dropblock_batch
           x = dropblock_batch(x, drop_prob=0.1, block_size=7, batchwise=True)
    """
    block_size = min((block_size,) + x.shape[2:])

    space = reduce(mul, x.shape[2:])
    valid_space = reduce(mul, map(lambda d: d - block_size + 1, x.shape[2:]))
    gamma = drop_prob / block_size**2 * space / valid_space

    mask_shape = (1,) + x.shape[1:] if batchwise else x.shape
    drop = torch.rand(mask_shape, dtype=x.dtype, device=x.device) < gamma
    drop = F.max_pool2d(drop.to(x.dtype), block_size, 1, block_size // 2)
    if not block_size % 2:
        drop = drop[list(map(slice, x.shape))]

    keep = 1 - drop
    keep_frac = keep.sum() / keep.numel() or 1
    return x * keep / keep_frac


class DropBlock(Algorithm):
    """`DropBlock <https://arxiv.org/abs/1810.12890>`_ is a form of structured dropout, where units in a contiguous
    region of a feature map are dropped together.

    Args:
        drop_prob (float): Drop probability.
        block_size (int): Size of blocks to drop out.
        batchwise (bool): Whether to mask per batch (faster) or per sample.
    """

    def __init__(self, drop_prob: float = 0.1, block_size: int = 7, batchwise: bool = True):
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.batchwise = batchwise

    def match(self, event: Event, state: State) -> bool:
        """Runs on Event.AFTER_DATALOADER."""
        return event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Apply dropblock on input tensor."""
        dur = state.get_elapsed_duration()
        fadein = np.tanh(dur.value * 2)
        fadein_drop_prob = fadein * self.drop_prob
        x, y = state.batch
        assert isinstance(x, Tensor), 'Multiple tensors not supported for DropBlock.'
        x2 = dropblock_batch(x, fadein_drop_prob, self.block_size, self.batchwise)
        state.batch = x2, y
