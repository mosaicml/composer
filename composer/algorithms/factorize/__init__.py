# Copyright 2021 MosaicML. All Rights Reserved.
from composer.algorithms.factorize.factorize import Factorize as Factorize
from composer.algorithms.factorize.factorize import FactorizeHparams as FactorizeHparams
from composer.algorithms.factorize.factorize import apply_factorization as apply_factorization
from composer.algorithms.factorize.factorize_core import LowRankSolution as LowRankSolution
from composer.algorithms.factorize.factorize_core import factorize_conv2d as factorize_conv2d
from composer.algorithms.factorize.factorize_core import factorize_matrix as factorize_matrix
from composer.algorithms.factorize.factorize_modules import FactorizedConv2d as FactorizedConv2d
from composer.algorithms.factorize.factorize_modules import FactorizedLinear as FactorizedLinear

_name = 'Factorize'
_class_name = 'Factorize'
_functional = 'apply_factorization'
_tldr = 'Factorize linear transforms into two smaller transforms'
_attribution = ''
_link = ''
_method_card = ''
