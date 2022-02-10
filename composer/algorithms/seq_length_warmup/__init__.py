# Copyright 2021 MosaicML. All Rights Reserved.

"""Sequence length warmup progressively increases the sequence length during training of
NLP models. See the :doc:`Method Card </method_cards/seq_len_warmup>` for more
details.
"""

from composer.algorithms.seq_length_warmup.seq_length_warmup import SeqLengthWarmup as SeqLengthWarmup
from composer.algorithms.seq_length_warmup.seq_length_warmup import \
    set_batch_sequence_length as set_batch_sequence_length

_name = 'Sequential Length Warmup'
_class_name = 'SeqLengthWarmup'
_functional = 'apply_seq_length_warmup'
_tldr = 'Progressively increase sequence length.'
_attribution = '(Li et al, 2021)'
_link = 'https://arxiv.org/abs/2108.06084'
_method_card = ''

__all__ = ["SeqLengthWarmup", "set_batch_sequence_length"]
