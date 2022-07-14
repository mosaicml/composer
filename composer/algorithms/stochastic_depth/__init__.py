# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Implements stochastic depth (`Huang et al, 2016 <https://arxiv.org/abs/1603.09382>`_) for ResNet blocks.

See :class:`.StochasticDepth`, the sample-wise stochastic depth :doc:`method card
</method_cards/stochastic_depth_samplewise>`, or the block-wise stochastic depth :doc:`method card
</method_cards/stochastic_depth>` for details.
"""

from composer.algorithms.stochastic_depth.stochastic_depth import StochasticDepth as StochasticDepth
from composer.algorithms.stochastic_depth.stochastic_depth import apply_stochastic_depth as apply_stochastic_depth

__all__ = ['StochasticDepth', 'apply_stochastic_depth']
