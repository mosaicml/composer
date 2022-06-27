# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Create new samples using convex combinations of pairs of samples.

This is done by taking a convex combination of x with a randomly permuted copy of x.

See the :doc:`Method Card </method_cards/mixup>` for more details.
"""

from composer.algorithms.mixup.mixup import MixUp as MixUp
from composer.algorithms.mixup.mixup import mixup_batch as mixup_batch

__all__ = ['MixUp', 'mixup_batch']
