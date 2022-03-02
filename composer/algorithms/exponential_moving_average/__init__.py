# Copyright 2021 MosaicML. All Rights Reserved.

"""Exponential moving average maintains a moving average of model parameters and uses these at test time.

See the :doc:`Method Card </method_cards/exponential_moving_average>` for more details.
"""

from composer.algorithms.exponential_moving_average.exponential_moving_average import ExponentialMovingAverage as ExponentialMovingAverage
from composer.algorithms.exponential_moving_average.exponential_moving_average import exponential_moving_average as exponential_moving_average

__all__ = ["ExponentialMovingAverage", "exponential_moving_average"]
