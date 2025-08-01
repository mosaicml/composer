# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import types

from tests.common.compare import deep_compare
from tests.common.datasets import (
    InfiniteClassificationDataset,
    ParityDataset,
    RandomClassificationDataset,
    RandomClassificationDatasetReplicated,
    RandomImageDataset,
    RandomSegmentationDataset,
    RandomTextClassificationDataset,
    RandomTextLMDataset,
    SimpleDataset,
)
from tests.common.events import EventCounterCallback
from tests.common.markers import device, world_size
from tests.common.models import (
    ComposerCounterModel,
    ConvModel,
    CountModule,
    EmbeddedWeightTiedModel,
    EmptyModel,
    EvenSimplerMLP,
    NestedFSDPModel,
    PartialWeightTiedModel,
    SimpleComposerMLP,
    SimpleConvModel,
    SimpleMLP,
    SimpleModel,
    SimpleModelWithDropout,
    SimpleTransformerClassifier,
    SimpleTransformerMaskedLM,
    SimpleWeightTiedModel,
    TPSimpleComposerMLP,
    ZeroModel,
    composer_resnet,
)
from tests.common.state import assert_state_equivalent


def get_module_subclasses(module: types.ModuleType, cls: type) -> list[type]:
    """Get all implementations of a class in a __module__ by scanning the re-exports from __init__.py"""
    return [x for x in vars(module).values() if isinstance(x, type) and issubclass(x, cls) and x is not cls]


__all__ = [
    'assert_state_equivalent',
    'RandomClassificationDataset',
    'RandomClassificationDatasetReplicated',
    'RandomTextClassificationDataset',
    'RandomTextLMDataset',
    'RandomImageDataset',
    'RandomSegmentationDataset',
    'ConvModel',
    'SimpleConvModel',
    'ZeroModel',
    'EmptyModel',
    'SimpleModel',
    'SimpleTransformerClassifier',
    'SimpleTransformerMaskedLM',
    'EmbeddedWeightTiedModel',
    'PartialWeightTiedModel',
    'NestedFSDPModel',
    'SimpleWeightTiedModel',
    'EventCounterCallback',
    'deep_compare',
    'device',
    'world_size',
    'get_module_subclasses',
    'SimpleModelWithDropout',
    'ParityDataset',
    'SimpleDataset',
    'InfiniteClassificationDataset',
    'composer_resnet',
    'SimpleMLP',
    'EvenSimplerMLP',
    'SimpleComposerMLP',
    'TPSimpleComposerMLP',
    'ComposerCounterModel',
    'CountModule',
]
