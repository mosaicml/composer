# Copyright 2021 MosaicML. All Rights Reserved.

"""Replaces model with a dummy model of type :class:`NoOpModelClass` on Event.INIT
"""

from composer.algorithms.no_op_model.no_op_model import NoOpModel as NoOpModel

__all__ = ['NoOpModel']
