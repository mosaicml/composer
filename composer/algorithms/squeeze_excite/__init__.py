# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Adds Squeeze-and-Excitation blocks (`Hu et al, 2019 <https://arxiv.org/abs/1709.01507>`_) after the
:class:`~torch.nn.Conv2d` modules in a neural network.

See :class:`~composer.algorithms.SqueezeExcite` or the :doc:`Method Card </method_cards/squeeze_excite>` for details.
"""

from composer.algorithms.squeeze_excite.squeeze_excite import SqueezeExcite as SqueezeExcite
from composer.algorithms.squeeze_excite.squeeze_excite import SqueezeExcite2d as SqueezeExcite2d
from composer.algorithms.squeeze_excite.squeeze_excite import SqueezeExciteConv2d as SqueezeExciteConv2d
from composer.algorithms.squeeze_excite.squeeze_excite import apply_squeeze_excite as apply_squeeze_excite

__all__ = ['SqueezeExcite', 'SqueezeExcite2d', 'SqueezeExciteConv2d', 'apply_squeeze_excite']
