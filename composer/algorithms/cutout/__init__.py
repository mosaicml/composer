# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.cutout.cutout import CutOut as CutOut
from composer.algorithms.cutout.cutout import CutOutHparams as CutOutHparams
from composer.algorithms.cutout.cutout import cutout as cutout

_name = 'CutOut'
_class_name = 'CutOut'
_functional = 'cutout'
_tldr = 'Randomly erases rectangular blocks from the image.'
_attribution = '(DeVries et al, 2017)'
_link = 'https://arxiv.org/abs/1708.04552'
_method_card = ''
