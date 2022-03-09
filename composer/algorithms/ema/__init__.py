# Copyright 2021 MosaicML. All Rights Reserved.

"""Exponential moving average maintains a moving average of model parameters and uses these at test time.

See the :doc:`Method Card </method_cards/exponential_moving_average>` for more details.
"""

from composer.algorithms.ema.ema import EMA as EMA
from composer.algorithms.ema.ema import ema as ema

__all__ = ["EMA", "ema"]
