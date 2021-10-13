# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.alibi import gpt2_alibi as gpt2_alibi
from composer.algorithms.alibi.alibi import Alibi as Alibi
from composer.algorithms.alibi.alibi import AlibiHparams as AlibiHparams
from composer.algorithms.alibi.alibi import apply_alibi as apply_alibi

_name = 'Alibi'
_class_name = 'Alibi'
_functional = 'apply_alibi'
_tldr = 'Replace attention with AliBi'
_attribution = '(Press et al, 2021)'
_link = 'https://arxiv.org/abs/2108.12409v1'
_method_card = ''
