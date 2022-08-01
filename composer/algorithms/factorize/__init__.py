# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Decomposes linear operators into pairs of smaller linear operators.

See :class:`.Factorize` or the :doc:`Method Card </method_cards/factorize>` for details.
"""

from composer.algorithms.factorize.factorize import Factorize as Factorize
from composer.algorithms.factorize.factorize import apply_factorization as apply_factorization
from composer.algorithms.factorize.factorize_core import LowRankSolution as LowRankSolution
from composer.algorithms.factorize.factorize_core import factorize_conv2d as factorize_conv2d
from composer.algorithms.factorize.factorize_core import factorize_matrix as factorize_matrix
from composer.algorithms.factorize.factorize_modules import FactorizedConv2d as FactorizedConv2d
from composer.algorithms.factorize.factorize_modules import FactorizedLinear as FactorizedLinear
from composer.algorithms.factorize.factorize_modules import factorizing_could_speedup as factorizing_could_speedup

__all__ = [
    'Factorize',
    'apply_factorization',
    'LowRankSolution',
    'factorize_conv2d',
    'factorize_matrix',
    'FactorizedConv2d',
    'FactorizedLinear',
    'factorizing_could_speedup',
]
