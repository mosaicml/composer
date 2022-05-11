# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Changes the memory format of the model to ``torch.channels_last``.

This usually improves GPU utilization. See the :doc:`Method Card </method_cards/channels_last>` for more details.
"""

from composer.algorithms.channels_last.channels_last import ChannelsLast as ChannelsLast
from composer.algorithms.channels_last.channels_last import apply_channels_last as apply_channels_last

__all__ = ['ChannelsLast', 'apply_channels_last']
