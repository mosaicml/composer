# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""SAM (`Foret et al, 2020 <https://arxiv.org/abs/2010.01412>`_) wraps an existing optimizer with a
:class:`SAMOptimizer` which makes the optimizer minimize both loss value and sharpness.This can improves model
generalization and provide robustness to label noise.

See the :doc:`Method Card </method_cards/sam>` for more details.
"""

from composer.algorithms.sam.sam import SAM as SAM
from composer.algorithms.sam.sam import SAMOptimizer as SAMOptimizer

__all__ = ['SAM', 'SAMOptimizer']
