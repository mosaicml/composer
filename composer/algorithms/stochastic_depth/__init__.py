# Copyright 2021 MosaicML. All Rights Reserved.

"""Adds Squeeze-and-Excitation blocks (`Hu et al, 2019 <https://arxiv.org/abs/1709.01507>`_) after the
:class:`~torch.nn.Conv2d` modules in a neural network.

See :class:`~composer.algorithms.StochasticDepth`, the sample-wise stochastic depth :doc:`method card
</method_cards/stochastic_depth_samplewise>`, or the block-wise stochastic depth :doc:`method card
</method_cards/stochastic_depth>` for details.
"""

from composer.algorithms.stochastic_depth.sample_stochastic_layers import \
    SampleStochasticBottleneck as SampleStochasticBottleneck
from composer.algorithms.stochastic_depth.stochastic_depth import StochasticDepth as StochasticDepth
from composer.algorithms.stochastic_depth.stochastic_depth import apply_stochastic_depth as apply_stochastic_depth
from composer.algorithms.stochastic_depth.stochastic_layers import StochasticBottleneck as StochasticBottleneck

__all__ = ["StochasticDepth", "apply_stochastic_depth", "StochasticBottleneck", "SampleStochasticBottleneck"]
