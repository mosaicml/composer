# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""This file provides the canonical settings (dataset, model, algorithms, arguments)
for each algorithm to be tested. This can be used throughout the codebase for
functional tests, serialization tests, etc.

Each algorithm is keyed based on its name in the algorithm registry.
"""

from typing import Any, Dict, Optional, Type

import pytest
from torch.utils.data import DataLoader, Dataset

import composer
import composer.algorithms
from composer import Algorithm
from composer.algorithms import (EMA, SAM, SWA, Alibi, AugMix, BlurPool, ChannelsLast, ColOut, CutMix, CutOut,
                                 Factorize, FusedLayerNorm, GatedLinearUnits, GhostBatchNorm, GradientClipping,
                                 GyroDropout, LabelSmoothing, LayerFreezing, LowPrecisionLayerNorm, MixUp, NoOpModel,
                                 ProgressiveResizing, RandAugment, SelectiveBackprop, SeqLengthWarmup, SqueezeExcite,
                                 StochasticDepth, WeightStandardization)
from composer.models import composer_resnet
from composer.models.base import ComposerModel
from tests.common import get_module_subclasses
from tests.common.datasets import RandomImageDataset, SimpleDataset, dummy_bert_lm_dataloader, dummy_gpt_lm_dataloader
from tests.common.models import (SimpleConvModel, SimpleModelWithDropout, configure_tiny_bert_hf_model,
                                 configure_tiny_gpt2_hf_model)

simple_bert_settings = {
    'model': configure_tiny_bert_hf_model,
    'dataloader': (dummy_bert_lm_dataloader, {
        'size': 8
    }),
    'kwargs': {},
}

simple_gpt2_settings = {
    'model': configure_tiny_gpt2_hf_model,
    'dataloader': (dummy_gpt_lm_dataloader, {
        'size': 8
    }),
    'kwargs': {},
}

simple_vision_settings = {
    'model': SimpleConvModel,
    'dataset': RandomImageDataset,
    'kwargs': {},
}

simple_vision_pil_settings = {
    'model': SimpleConvModel,
    'dataset': (RandomImageDataset, {
        'is_PIL': True
    }),
    'kwargs': {},
}

simple_resnet_settings = {
    'model': (composer_resnet, {
        'model_name': 'resnet18',
        'num_classes': 2
    }),
    'dataset': (RandomImageDataset, {
        'shape': (3, 224, 224),
    }),
    'kwargs': {},
}

_settings: Dict[Type[Algorithm], Optional[Dict[str, Any]]] = {
    GradientClipping: {
        'model': SimpleConvModel,
        'dataset': RandomImageDataset,
        'kwargs': {
            'clipping_type': 'norm',
            'clipping_threshold': 0.1
        },
    },
    Alibi: {
        'model': configure_tiny_bert_hf_model,
        'dataloader': (dummy_bert_lm_dataloader, {
            'size': 8
        }),
        'kwargs': {
            'max_sequence_length': 256
        },
    },
    AugMix: simple_vision_settings,
    BlurPool: {
        'model': SimpleConvModel,
        'dataset': RandomImageDataset,
        'kwargs': {
            'min_channels': 0,
        },
    },
    ChannelsLast: simple_vision_settings,
    ColOut: simple_vision_settings,
    CutMix: {
        'model': SimpleConvModel,
        'dataset': RandomImageDataset,
        'kwargs': {}
    },
    CutOut: simple_vision_settings,
    EMA: {
        'model': SimpleConvModel,
        'dataset': RandomImageDataset,
        'kwargs': {
            'half_life': '1ba',
        },
    },
    Factorize: simple_resnet_settings,
    FusedLayerNorm: simple_bert_settings,
    GatedLinearUnits: simple_bert_settings,
    GhostBatchNorm: {
        'model': (composer_resnet, {
            'model_name': 'resnet18',
            'num_classes': 2
        }),
        'dataset': (RandomImageDataset, {
            'shape': (3, 224, 224)
        }),
        'kwargs': {
            'ghost_batch_size': 2,
        }
    },
    LabelSmoothing: simple_vision_settings,
    LayerFreezing: simple_vision_settings,
    LowPrecisionLayerNorm: simple_bert_settings,
    MixUp: simple_vision_settings,
    ProgressiveResizing: simple_vision_settings,
    RandAugment: simple_vision_settings,
    NoOpModel: simple_vision_settings,
    SAM: simple_vision_settings,
    SelectiveBackprop: simple_vision_settings,
    SeqLengthWarmup: {
        'model': configure_tiny_bert_hf_model,
        'dataloader': (dummy_bert_lm_dataloader, {
            'size': 8
        }),
        'kwargs': {
            'duration': 0.5,
            'min_seq_length': 8,
            'max_seq_length': 16
        },
    },
    SqueezeExcite: simple_resnet_settings,
    StochasticDepth: {
        'model': (composer_resnet, {
            'model_name': 'resnet50',
            'num_classes': 2
        }),
        'dataset': (RandomImageDataset, {
            'shape': (3, 224, 224),
        }),
        'kwargs': {
            'stochastic_method': 'block',
            'target_layer_name': 'ResNetBottleneck',
            'drop_rate': 0.2,
            'drop_distribution': 'linear',
            'drop_warmup': '0.0dur',
        }
    },
    SWA: {
        'model': SimpleConvModel,
        'dataset': RandomImageDataset,
        'kwargs': {
            'swa_start': '0.2dur',
            'swa_end': '0.97dur',
            'update_interval': '1ep',
            'schedule_swa_lr': True,
        }
    },
    WeightStandardization: simple_vision_settings,
    GyroDropout: {
        'model': SimpleModelWithDropout,
        'dataloader': (DataLoader, {
            'dataset': SimpleDataset(batch_size=2, feature_size=64, num_classes=10)
        }),
        'kwargs': {
            'p': 0.5,
            'sigma': 2,
            'tau': 1
        }
    },
}


def _get_alg_settings(alg_cls: Type[Algorithm]):
    if alg_cls not in _settings or _settings[alg_cls] is None:
        raise ValueError(f'Algorithm {alg_cls.__name__} not in the settings dictionary.')
    settings = _settings[alg_cls]
    assert settings is not None
    return settings


def get_alg_kwargs(alg_cls: Type[Algorithm]) -> Dict[str, Any]:
    """Return the kwargs for an algorithm."""
    return _get_alg_settings(alg_cls)['kwargs']


def get_alg_model(alg_cls: Type[Algorithm]) -> ComposerModel:
    """Return an instance of the model for an algorithm."""
    settings = _get_alg_settings(alg_cls)['model']
    if isinstance(settings, tuple):
        (cls, kwargs) = settings
    else:
        (cls, kwargs) = (settings, {})
    return cls(**kwargs)


def get_alg_dataloader(alg_cls: Type[Algorithm]) -> DataLoader:
    """Return an instance of the dataset for an algorithm."""
    settings = _get_alg_settings(alg_cls)

    if 'dataloader' in settings:
        settings = settings['dataloader']
    elif 'dataset' in settings:
        settings = settings['dataset']
    else:
        raise ValueError(f'Neither dataset nor dataloader have been provided for algorithm {alg_cls}')

    if isinstance(settings, tuple):
        (cls, kwargs) = settings
    else:
        (cls, kwargs) = (settings, {})

    dataloader = cls(**kwargs)
    if isinstance(dataloader, Dataset):
        dataloader = DataLoader(dataset=dataloader, batch_size=2)
    return dataloader


def get_algs_with_marks():
    """Returns a list of algorithms appropriate markers for a subsequent call to pytest.mark.parameterize.
    It applies markers as appropriate (e.g. XFAIL for algs missing config)
    It reads from the algorithm registry

    E.g. @pytest.mark.parametrize("alg_class", get_algs_with_marks())
    """
    ans = []
    for alg_cls in get_module_subclasses(composer.algorithms, Algorithm):
        marks = []
        settings = _settings[alg_cls]

        if alg_cls in (Alibi, GatedLinearUnits, SeqLengthWarmup):
            try:
                import transformers
                transformers_available = True
                del transformers
            except ImportError:
                transformers_available = False
            marks.append(pytest.mark.skipif(not transformers_available, reason='transformers not available'))

        if alg_cls == SWA:
            # TODO(matthew): Fix
            marks.append(
                pytest.mark.filterwarnings(
                    r'ignore:Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`:UserWarning'))

        if alg_cls == MixUp:
            # TODO(Landen): Fix
            marks.append(
                pytest.mark.filterwarnings(r'ignore:Some targets have less than 1 total probability:UserWarning'))

        if alg_cls == FusedLayerNorm:
            # FusedLayerNorm requires a GPU in order for the class to exist
            marks.append(pytest.mark.gpu)

        if settings is None:
            marks.append(pytest.mark.xfail(reason=f'Algorithm {alg_cls.__name__} is missing settings.'))

        ans.append(pytest.param(alg_cls, marks=marks, id=alg_cls.__name__))

    return ans
