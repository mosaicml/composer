# Copyright 2021 MosaicML. All Rights Reserved.

"""Shrinks targets towards a uniform distribution to counteract label noise as in `Szegedy et al
<https://arxiv.org/abs/1512.00567>`_. Introduced in `Rethinking the Inception Architecture for
Computer Vision <https://arxiv.org/abs/1512.00567>`_. See the :doc:`Method Card </method_cards/label_smoothing>`
for more details.
"""

from composer.algorithms.label_smoothing.label_smoothing import LabelSmoothing as LabelSmoothing
from composer.algorithms.label_smoothing.label_smoothing import smooth_labels as smooth_labels

_name = 'Label Smoothing'
_class_name = 'LabelSmoothing'
_functional = 'smooth_labels'
_tldr = 'Smooths the labels with a uniform prior'
_attribution = '(Szegedy et al, 2015)'
_link = 'https://arxiv.org/abs/1512.00567'
_method_card = 'docs/source/method_cards/label_smoothing.md'

__all__ = ["LabelSmoothing", "smooth_labels"]