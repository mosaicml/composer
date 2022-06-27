# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Progressively freeze the layers of the network during training, starting with the earlier layers.

See the :doc:`Method Card </method_cards/layer_freezing>` for more details.
"""

from composer.algorithms.layer_freezing.layer_freezing import LayerFreezing as LayerFreezing
from composer.algorithms.layer_freezing.layer_freezing import freeze_layers as freeze_layers

__all__ = ['LayerFreezing', 'freeze_layers']
