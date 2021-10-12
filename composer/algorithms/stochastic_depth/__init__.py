# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.stochastic_depth.sample_stochastic_layers import \
    SampleStochasticBottleneck as SampleStochasticBottleneck
from composer.algorithms.stochastic_depth.stochastic_depth import StochasticDepth as StochasticDepth
from composer.algorithms.stochastic_depth.stochastic_depth import StochasticDepthHparams as StochasticDepthHparams
from composer.algorithms.stochastic_depth.stochastic_depth import apply_stochastic_depth as apply_stochastic_depth
from composer.algorithms.stochastic_depth.stochastic_layers import StochasticBottleneck as StochasticBottleneck

_name = 'Stochastic Depth'
_class_name = 'StochasticDepth'
_functional = 'apply_stochastic_depth'
_tldr = 'Replaces a specified layer with a stochastic verion that randomly drops the layer or samples during training'
_attribution = '(Huang et al, 2016)'
_link = 'https://arxiv.org/abs/1603.09382'
_method_card = ''
