# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.selective_backprop.selective_backprop import SelectiveBackprop as SelectiveBackprop
from composer.algorithms.selective_backprop.selective_backprop import \
    SelectiveBackpropHparams as SelectiveBackpropHparams
from composer.algorithms.selective_backprop.selective_backprop import selective_backprop as selective_backprop

_name = 'Selective Backprop'
_class_name = 'SelectiveBackprop'
_functional = 'selective_backprop'
_tldr = 'Drops examples with small loss contributions.'
_attribution = '(Jiang et al, 2019)'
_link = 'https://arxiv.org/abs/1910.00762'
_method_card = ''
