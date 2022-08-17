# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`Inspired by Selective Backprop <https://arxiv.org/abs/1910.00762>`_: prunes minibatches according to the difficulty of the
individual training examples, and only computes weight gradients over the pruned subset, reducing iteration time and
speeding up training.

The algorithm runs on :attr:`~composer.core.event.Event.INIT` and :attr:`~composer.core.event.Event.AFTER_DATLOADER`.
On Event.INIT, it gets the loss function before the model is wrapped. On Event.AFTER_DATALOADER, it applies selective
backprop if the time is between ``self.start`` and ``self.end``.

"""

import importlib
import os
import sys
from pathlib import Path
from typing import Callable

from composer.algorithms.sample_prioritization.sample_prioritization import (SamplePrioritization, select_using_loss,
                                                                             should_selective_backprop)

__all__ = ['SamplePrioritization', 'select_using_loss', 'should_selective_backprop']
