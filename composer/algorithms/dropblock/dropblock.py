# Copyright 2021 MosaicML. All Rights Reserved.

import logging
from dataclasses import asdict, dataclass
from functools import reduce
from operator import mul
from typing import Optional

import torch
import yahp as hp
from torch.nn import functional as F

from composer.algorithms import AlgorithmHparams
from composer.core.types import Algorithm, Event, Logger, State, Tensor

log = logging.getLogger(__name__)


def dropblock(x: Tensor, drop_prob: float = 0.1, block_size: int = 7, batchwise: bool = True):
    '''
    See :class:`DropBlock`.

    Args:
        drop_prob (float): Drop probability
        block_size (int): Size of blocks to drop out
        batchwise (bool): Whether to mask per batch (faster) or per sample
    '''
    block_size = min((block_size,) + x.shape[2:])

    space = reduce(mul, x.shape[2:])
    valid_space = reduce(mul, map(lambda d: d - block_size + 1, x.shape[2:]))
    gamma = drop_prob / block_size**2 * space / valid_space

    mask_shape = (1,) + x.shape[1:] if batchwise else x.shape
    mask = torch.rand(mask_shape, dtype=x.dtype, device=x.device) < gamma
    mask = F.max_pool2d(mask.to(x.dtype), block_size, 1, block_size // 2)
    mask = 1 - mask

    mask_frac = mask.sum() / mask.numel()
    return x * mask / mask_frac


@dataclass
class DropBlockHparams(AlgorithmHparams):
    '''See :class:`DropBlock`.'''

    drop_prob: float = hp.required('Drop probability', template_default=0.1)
    block_size: int = hp.required('Size of blocks to drop out', template_default=7)
    batchwise: bool = hp.required('Whether to mask per batch (faster) or per sample', template_default=True)

    def initialize_object(self) -> 'DropBlock':
        return DropBlock(**asdict(self))


class DropBlock(Algorithm):
    '''`DropBlock <https://arxiv.org/abs/1810.12890>`_ is a form of structured dropout, where units
    in a contiguous region of a feature map are dropped together.

    Args:
        drop_prob (float): Drop probability
        block_size (int): Size of blocks to drop out
        batchwise (bool): Whether to mask per batch (faster) or per sample
    '''

    def __init__(self, drop_prob: float, block_size: int, batchwise: bool):
        self.hparams = DropBlockHparams(drop_prob, block_size, batchwise)

    def match(self, event: Event, state: State) -> bool:
        return event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        x, y = state.batch_pair
        assert isinstance(x, Tensor), 'Multiple tensors not supported for DropBlock.'
        new_x = dropblock(x, **asdict(self.hparams))
        state.batch = new_x, y
