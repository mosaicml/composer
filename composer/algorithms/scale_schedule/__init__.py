# Copyright 2021 MosaicML. All Rights Reserved.

"""Deprecated - do not use. Currently does not make any changes to the trainer. 
Instead, use the ``scale_schedule_ratio`` parameter of the Composer Trainer.
"""

from composer.algorithms.scale_schedule.scale_schedule import ScaleSchedule as ScaleSchedule

__all__ = ['ScaleSchedule']
