# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.ghost_batchnorm.ghost_batchnorm import GhostBatchNorm as GhostBatchNorm
from composer.algorithms.ghost_batchnorm.ghost_batchnorm import GhostBatchNormHparams as GhostBatchNormHparams
from composer.algorithms.ghost_batchnorm.ghost_batchnorm import apply_ghost_batchnorm as apply_ghost_batchnorm

_name = 'Ghost BatchNorm'
_class_name = 'GhostBatchNorm'
_functional = 'apply_ghost_batchnorm'
_tldr = 'Use smaller samples to compute batchnorm'
_attribution = '(Dimitriou et al, 2020)'
_link = 'https://arxiv.org/abs/2007.08554'
_method_card = ''
