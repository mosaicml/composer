# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Replaces batch normalization modules with `Ghost Batch Normalization <https://arxiv.org/abs/1705.08741>`_ modules
that simulate the effect of using a smaller batch size.

See :class:`~composer.algorithms.GhostBatchNorm` or the :doc:`Method Card </method_cards/ghost_batchnorm>` for details.
"""

from composer.algorithms.ghost_batchnorm.ghost_batchnorm import GhostBatchNorm as GhostBatchNorm
from composer.algorithms.ghost_batchnorm.ghost_batchnorm import apply_ghost_batchnorm as apply_ghost_batchnorm

__all__ = ['GhostBatchNorm', 'apply_ghost_batchnorm']
