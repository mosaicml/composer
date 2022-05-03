# Copyright 2021 MosaicML. All Rights Reserved.

"""Exponential moving average maintains a moving average of model parameters and uses these at test time.

See the :doc:`Method Card </method_cards/ema>` for more details.
"""

from composer.algorithms.ema.ema import EMA as EMA
from composer.algorithms.ema.ema import compute_ema as compute_ema

__all__ = ["EMA", "compute_ema"]
