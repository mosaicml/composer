# Copyright 2021 MosaicML. All Rights Reserved.

from torch import Tensor, nn

from composer.algorithms.dropblock.dropblock import dropblock


class DropBlock(nn.Module):
    '''`Dropblock <https://arxiv.org/abs/1810.12890>`_ is a form of structured dropout, where units
    in a contiguous region of a feature map are dropped together.

    This module is like a Dropout for Conv layers.

    Args:
        drop_prob (float, optional): Drop probability. (default: 0.1)
        block_size (int, optional): Size of blocks to drop out. (default: 7)
        batchwise (bool, optional): Whether to reuse masks across batch (faster). (default: True)
    '''

    def __init__(self, drop_prob: float = 0.1, block_size: int = 7, batchwise: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.batchwise = batchwise

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or not self.drop_prob:
            return x
        return dropblock(x, self.drop_prob, self.block_size, self.batchwise)
