# Copyright 2021 MosaicML. All Rights Reserved.

"""Sequence length warmup progressively increases the sequence length during training of
NLP models.
"""

from composer.algorithms.seq_length_warmup.seq_length_warmup import SeqLengthWarmup as SeqLengthWarmup
from composer.algorithms.seq_length_warmup.seq_length_warmup import SeqLengthWarmupHparams as SeqLengthWarmupHparams
from composer.algorithms.seq_length_warmup.seq_length_warmup import apply_seq_length_warmup as apply_seq_length_warmup

_name = 'Sequential Length Warmup'
_class_name = 'SeqLengthWarmup'
_functional = 'apply_seq_length_warmup'
_tldr = 'Progressively increase sequence length.'
_attribution = '(Li et al, 2021)'
_link = 'https://arxiv.org/abs/2108.06084'
_method_card = ''

__all__ = ["SeqLengthWarmup", "SeqLengthWarmupHparams", "apply_seq_length_warmup"]
