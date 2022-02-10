# Copyright 2021 MosaicML. All Rights Reserved.

"""ALiBi (Attention with Linear Biases) dispenses with position
embeddings for tokens in transformer-based NLP models, instead encoding position
information by biasing the query-key attention scores proportionally to each token pair's
distance. Introduced in `Train Short, Test Long: Attention with Linear Biases Enables
Input Length Extrapolation <https://arxiv.org/abs/2108.12409>`_ (Press et al., 2021). See
the :doc:`Method Card <docs/source/method_cards/alibi.md>` for more details.
"""

from composer.algorithms.alibi import _gpt2_alibi as _gpt2_alibi
from composer.algorithms.alibi.alibi import Alibi as Alibi
from composer.algorithms.alibi.alibi import AlibiHparams as AlibiHparams
from composer.algorithms.alibi.alibi import apply_alibi as apply_alibi

_name = 'Alibi'
_class_name = 'Alibi'
_functional = 'apply_alibi'
_tldr = 'Replace attention with AliBi'
_attribution = '(Press et al, 2021)'
_link = 'https://arxiv.org/abs/2108.12409v1'
_method_card = 'docs/source/method_cards/alibi.md'

__all__ = ["Alibi", "AlibiHparams", "apply_alibi"]
