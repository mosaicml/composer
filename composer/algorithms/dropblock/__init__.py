# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.dropblock.dropblock import DropBlock as DropBlock
from composer.algorithms.dropblock.dropblock import DropBlockHparams as DropBlockHparams
from composer.algorithms.dropblock.dropblock import dropblock as dropblock
from composer.algorithms.dropblock.dropblock_layers import DropBlock as DropBlockNd

_name = 'DropBlock'
_class_name = 'DropBlock'
_functional = 'dropblock'
_tldr = 'Structured dropout, where units in a contiguous region of a feature map are dropped together'
_attribution = '(Ghiasi et al, 2018)'
_link = 'https://arxiv.org/abs/1810.12890'
_method_card = ''
