# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import types
from typing import List, Type

from tests.common.compare import deep_compare
from tests.common.datasets import (InfiniteClassificationDataset, RandomClassificationDataset, RandomImageDataset,
                                   RandomSegmentationDataset, RandomTextClassificationDataset, SimpleDataset)
from tests.common.events import EventCounterCallback
from tests.common.markers import device, world_size
from tests.common.models import (ConvModel, EmbeddedWeightTiedModel, SimpleConvModel, SimpleModel,
                                 SimpleModelWithDropout, SimpleTransformerClassifier, SimpleWeightTiedModel)
from tests.common.state import assert_state_equivalent


def get_module_subclasses(module: types.ModuleType, cls: Type) -> List[Type]:
    """Get all implementations of a class in a __module__ by scanning the re-exports from __init__.py"""
    return [x for x in vars(module).values() if isinstance(x, type) and issubclass(x, cls) and x is not cls]


__all__ = [
    'assert_state_equivalent',
    'RandomClassificationDataset',
    'RandomTextClassificationDataset',
    'RandomImageDataset',
    'RandomSegmentationDataset',
    'ConvModel',
    'SimpleConvModel',
    'SimpleModel',
    'SimpleTransformerClassifier',
    'EmbeddedWeightTiedModel',
    'SimpleWeightTiedModel',
    'EventCounterCallback',
    'deep_compare',
    'device',
    'world_size',
    'get_module_subclasses',
    'SimpleModelWithDropout',
    'SimpleDataset',
    'InfiniteClassificationDataset',
]
