# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.swa.hparams import SWAHparams as SWAHparams
from composer.algorithms.swa.swa import SWA as SWA

"""Stochastic Weight Averaging (SWA) averages model weights sampled at different times
near the end of training, leading to better generalization than just using the final
trained weights.
"""

_name = 'SWA'
_class_name = 'SWA'
_functional = ''
_tldr = 'Computes running average of model weights.'
_attribution = '(Izmailov et al, 2018)'
_link = 'https://arxiv.org/abs/1803.05407'
_method_card = ''

__all__ = ['SWA', 'SWAHparams']
