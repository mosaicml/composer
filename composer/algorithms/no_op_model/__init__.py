# Copyright 2021 MosaicML. All Rights Reserved.

"""Replaces model with a dummy model of type :class:`NoOpModelClass`.

The algorithm runs on Event.INIT. It replaces the model in the state with 
a :class:`NoOpModelClass` and then updates the parameters in the optimizer
through module surgery.

Using a dummy model can help profiling the dataloader by eliminating the work
necessary to compute model outputs.
"""

from composer.algorithms.no_op_model.no_op_model import NoOpModel as NoOpModel
from composer.algorithms.no_op_model.no_op_model import NoOpModelClass as NoOpModelClass

__all__ = ['NoOpModel', 'NoOpModelClass']
