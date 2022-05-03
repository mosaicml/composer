"""This file provides the canonical settings (dataset, model, algorithms, arguments) for each algorithm to be tested.
This can be used throughout the codebase for functional tests, serialization tests, etc.

Each algorithm is keyed based on its name in the algorithm registry.
"""

from composer.algorithms import algorithm_registry
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
    'kwargs': {}
}

simple_resnet_settings = {
    'model': (ComposerResNet, {
        'model_name': 'resnet18',
        'num_classes': 2
    }),
    'dataset': (common.RandomImageDataset, {
        'shape': (3, 224, 224),
    }),
}

_settings = {
    'agc': simple_vision_settings,
    'alibi': None,  # NLP settings needed
    'augmix': None,  # requires PIL dataset to test
    'blurpool': {
        'model': common.SimpleConvModel,
        'dataset': common.RandomImageDataset,
        'kwargs': {
            'min_channels': 0,
        },
    },
    'channels_last': simple_vision_settings,
    'colout': simple_vision_settings,
    'cutmix': {
        'model': common.SimpleConvModel,
        'dataset': common.RandomImageDataset,
        'kwargs': {
            'num_classes': 2
        }
    },
    'cutout': simple_vision_settings,
    'ema': simple_vision_settings,
    'factorize': None,
    'ghost_batchnorm': {
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
    'label_smoothing': simple_vision_settings,
    'layer_freezing': simple_vision_settings,
    'mixup': simple_vision_settings,
    'progressive_resizing': simple_vision_settings,
    'randaugment': None,  # requires PIL dataset to test
    'sam': simple_vision_settings,
    'selective_backprop': simple_vision_settings,
    'seq_length_warmup': None,  # NLP settings needed
    'squeeze_excite': simple_resnet_settings,
    'stochastic_depth': {
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
            'use_same_gpu_seed': False
        }
    },
    'swa': {
        'model': common.SimpleConvModel,
        'dataset': common.RandomImageDataset,
        'kwargs': {
            'swa_start': "1ep",
            'swa_end': "0.97dur",
            'update_interval': '1ep',
            'schedule_swa_lr': True,
        }
    }
}


def get_settings(name: str):
    """For a given algorithm name, creates the canonical setting (algorithm, model, dataset) for testing.

    Returns ``None`` if no settings provided.
    """
    if name not in _settings:
        raise ValueError(f'No settings for {name} found, please add.')

    setting = _settings[name]
    if setting is None:
        return None

    result = {}
    for key in ('model', 'dataset'):
        if isinstance(setting[key], tuple):
            (obj, kwargs) = setting[key]
        else:
            (obj, kwargs) = (setting[key], {})

        # create the object
        result[key] = obj(**kwargs)

    # create algorithm
    kwargs = setting.get('kwargs', {})
    hparams = algorithm_registry.get_algorithm_registry()[name]
    result['algorithm'] = hparams(**kwargs).initialize_object()
    return result
