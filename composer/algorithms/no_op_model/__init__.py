# Copyright 2021 MosaicML. All Rights Reserved.

"""Replaces model with a dummy model of type :class:`NoOpModelClass`.

The algorithm runs on Event.INIT. It replaces the model in the state with 
a :class:`NoOpModelClass` and then updates the parameters in the optimizer
through module surgery.
"""

from composer.algorithms.no_op_model.no_op_model import NoOpModel as NoOpModel

__all__ = ['NoOpModel']
