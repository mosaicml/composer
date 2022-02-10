# Copyright 2021 MosaicML. All Rights Reserved.

"""Stochastic Weight Averaging (SWA; `Izmailov et al., 2018
<https://arxiv.org/abs/1803.05407>`_) averages model weights sampled at different times
near the end of training. This leads to better generalization than just using the final
trained weights.) averages model weights sampled at different times near the end of
training, leading to better generalization than just using the final trained weights. See
the :doc:`Method Card </method_cards/swa>` for more details.
"""

from composer.algorithms.swa.swa import SWA as SWA
from composer.algorithms.swa.swa import SWAHparams as SWAHparams

_name = 'SWA'
_class_name = 'SWA'
_functional = ''
_tldr = 'Computes running average of model weights.'
_attribution = '(Izmailov et al, 2018)'
_link = 'https://arxiv.org/abs/1803.05407'
_method_card = ''

__all__ = ['SWA', 'SWAHparams']
