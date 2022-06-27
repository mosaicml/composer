# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Shrinks targets towards a uniform distribution to counteract label noise. Introduced in `Rethinking the Inception
Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`_.

See the :doc:`Method Card </method_cards/label_smoothing>` for more details.
"""

from composer.algorithms.label_smoothing.label_smoothing import LabelSmoothing as LabelSmoothing
from composer.algorithms.label_smoothing.label_smoothing import smooth_labels as smooth_labels

__all__ = ['LabelSmoothing', 'smooth_labels']
