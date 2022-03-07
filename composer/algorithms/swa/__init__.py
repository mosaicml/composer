# Copyright 2021 MosaicML. All Rights Reserved.

"""Stochastic Weight Averaging (SWA; `Izmailov et al, 2018 <https://arxiv.org/abs/1803.05407>`_) averages model weights
sampled at different times near the end of training.

This leads to better generalization than just using the final trained weights. See the :doc:`Method Card
</method_cards/swa>` for more details.
"""

from composer.algorithms.swa.swa import SWA

__all__ = ['SWA']
