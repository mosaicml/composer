# Copyright 2021 MosaicML. All Rights Reserved.

"""`Automatic Gradient Clipping <https://arxiv.org/abs/2102.06171>`_ Clips all gradients 
in model based on ratio of gradient norms to parameter norms.

See the :doc:`Method Card </method_cards/agc>` for more details.
"""

from composer.algorithms.agc.agc import AGC
from composer.algorithms.agc.agc import apply_agc

__all__ = ["AGC", "apply_agc"]