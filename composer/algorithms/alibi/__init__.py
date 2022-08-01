# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""ALiBi (Attention with Linear Biases; `Press et al, 2021 <https://arxiv.org/abs/2108.12409>`_) dispenses with position
embeddings for tokens in transformer-based NLP models, instead encoding position information by biasing the query-key
attention scores proportionally to each token pair's distance.

See the :doc:`Method Card </method_cards/alibi>` for more details.
"""

from composer.algorithms.alibi.alibi import Alibi, apply_alibi

__all__ = ['Alibi', 'apply_alibi']
