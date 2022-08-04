# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from composer.algorithms import CopyPaste
from composer.algorithms.copypaste.copypaste import copypaste_batch
from composer.core import Event
from composer.models import ComposerClassifier
