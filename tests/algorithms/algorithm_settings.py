"""This file provides the canonical settings (dataset, model, algorithms, arguments)
for each algorithm to be tested. This can be used throughout the codebase for
functional tests, serialization tests, etc.

Each algorithm is keyed based on its name in the algorithm registry.
"""

from typing import Any, Dict, Optional, Type

import pytest

import composer
from composer.algorithms.agc import AGC
from composer.algorithms.alibi import Alibi
from composer.algorithms.augmix import AugMix
from composer.algorithms.blurpool import BlurPool
from composer.algorithms.channels_last import ChannelsLast
from composer.algorithms.colout import ColOut
from composer.algorithms.cutmix import CutMix
from composer.algorithms.cutout import CutOut
from composer.algorithms.ema import EMA
from composer.algorithms.factorize import Factorize
from composer.algorithms.ghost_batchnorm import GhostBatchNorm
from composer.algorithms.label_smoothing import LabelSmoothing
from composer.algorithms.layer_freezing import LayerFreezing
from composer.algorithms.mixup import MixUp
from composer.algorithms.no_op_model.no_op_model import NoOpModel
from composer.algorithms.progressive_resizing import ProgressiveResizing
from composer.algorithms.randaugment import RandAugment
from composer.algorithms.sam import SAM
from composer.algorithms.selective_backprop import SelectiveBackprop
from composer.algorithms.seq_length_warmup import SeqLengthWarmup
from composer.algorithms.squeeze_excite import SqueezeExcite
from composer.algorithms.stochastic_depth import StochasticDepth
from composer.algorithms.swa import SWA
from composer.core.algorithm import Algorithm
from composer.models import ComposerResNet
from tests import common

simple_vision_settings = {
    'model': common.SimpleConvModel,
    'dataset': common.RandomImageDataset,
    'kwargs': {},
}

simple_vision_pil_settings = {
    'model': common.SimpleConvModel,
    'dataset': (common.RandomImageDataset, {
        'is_PIL': True
    }),
    'kwargs': {},
}

simple_resnet_settings = {
    'model': (ComposerResNet, {
        'model_name': 'resnet18',
        'num_classes': 2
    }),
    'dataset': (common.RandomImageDataset, {
        'shape': (3, 224, 224),
    }),
    'kwargs': {},
}

_settings: Dict[Type[Algorithm], Optional[Dict[str, Any]]] = {
    AGC: simple_vision_settings,
    Alibi: None,  # NLP settings needed
    AugMix: None,  # requires PIL dataset to test
    BlurPool: {
        'model': common.SimpleConvModel,
        'dataset': common.RandomImageDataset,
        'kwargs': {
            'min_channels': 0,
        },
    },
    ChannelsLast: simple_vision_settings,
    ColOut: simple_vision_settings,
    CutMix: {
        'model': common.SimpleConvModel,
        'dataset': common.RandomImageDataset,
        'kwargs': {
            'num_classes': 2
        }
    },
    CutOut: simple_vision_settings,
    Factorize: simple_resnet_settings,
    EMA: {
        'model': common.SimpleConvModel,
        'dataset': common.RandomImageDataset,
        'kwargs': {
            'half_life': '100ba',
        },
    },
    GhostBatchNorm: {
        'model': (ComposerResNet, {
            'model_name': 'resnet18',
            'num_classes': 2
        }),
        'dataset': (common.RandomImageDataset, {
            'shape': (3, 224, 224)
        }),
        'kwargs': {
            'ghost_batch_size': 2,
        }
    },
    LabelSmoothing: simple_vision_settings,
    LayerFreezing: simple_vision_settings,
    MixUp: simple_vision_settings,
    ProgressiveResizing: simple_vision_settings,
    RandAugment: None,
    NoOpModel: None,
    SAM: simple_vision_settings,
    SelectiveBackprop: simple_vision_settings,
    SeqLengthWarmup: None,  # NLP settings needed
    SqueezeExcite: simple_resnet_settings,
    StochasticDepth: {
        'model': (ComposerResNet, {
            'model_name': 'resnet50',
            'num_classes': 2
        }),
        'dataset': (common.RandomImageDataset, {
            'shape': (3, 224, 224),
        }),
        'kwargs': {
            'stochastic_method': 'block',
            'target_layer_name': 'ResNetBottleneck',
            'drop_rate': 0.2,
            'drop_distribution': 'linear',
            'drop_warmup': "0.0dur",
            'use_same_gpu_seed': False,
        }
    },
    SWA: {
        'model': common.SimpleConvModel,
        'dataset': common.RandomImageDataset,
        'kwargs': {
            'swa_start': "1ep",
            'swa_end': "0.97dur",
            'update_interval': '1ep',
            'schedule_swa_lr': True,
        }
    },
}


def get_alg_kwargs(alg_cls: Type[Algorithm]):
    """Return the kwargs for an algorithm, or None if no settings exist."""
    if alg_cls not in _settings:
        raise ValueError(f"Algorithm {alg_cls.__name__} not in the settings dictionary.")
    settings = _settings[alg_cls]
    if settings is None:
        return None
    return settings['kwargs']


def get_algorithm_parametrization():
    """Returns a list of algorithms for a subsequent call to pytest.mark.parameterize.
    It applies markers as appropriate (e.g. GPU required for ChannelsLast, XFAIL for algs missing config, importskip for NLP algs)
    It reads from the algorithm registry
    
    E.g. @pytest.mark.parametrize("alg_class,alg_kwargs,model,dataset", get_algorithms_parametrization())
    """
    ans = []
    for alg_cls in common.get_all_subclasses_in_module(composer.algorithms, Algorithm):
        marks = []
        result = []
        settings = _settings[alg_cls]

        if alg_cls in (CutMix, MixUp, LabelSmoothing):
            # see: https://github.com/mosaicml/composer/issues/362
            pytest.importorskip("torch", minversion="1.10", reason="Pytorch 1.10 required.")

        result = [alg_cls]
        if settings is None:
            marks.append(pytest.mark.xfail(reason=f"Algorithm {alg_cls.__name__} is missing settings."))
            result.append({})  # no kwargs
            result.append(None)  # no model
            result.append(None)  # no dataset

        else:
            alg_kwargs = settings['kwargs']
            result.append(alg_kwargs)

            for key in ('model', 'dataset'):
                if isinstance(settings[key], tuple):
                    (obj, kwargs) = settings[key]
                else:
                    (obj, kwargs) = (settings[key], {})

                # create the object
                result.append(obj(**kwargs))

        ans.append(pytest.param(*result, marks=marks, id=alg_cls.__name__))

    return ans
