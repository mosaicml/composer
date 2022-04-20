# Copyright 2021 MosaicML. All Rights Reserved.

import dataclasses

import pytest

from composer.algorithms import (AlgorithmHparams, AlibiHparams, AugMixHparams, BlurPoolHparams, ChannelsLastHparams,
                                 ColOutHparams, CutMixHparams, CutOutHparams, FactorizeHparams, GhostBatchNormHparams,
                                 LabelSmoothingHparams, LayerFreezingHparams, MixUpHparams, NoOpModelHparams,
                                 ProgressiveResizingHparams, RandAugmentHparams, SAMHparams, ScaleScheduleHparams,
                                 SelectiveBackpropHparams, SeqLengthWarmupHparams, SqueezeExciteHparams,
                                 StochasticDepthHparams, SWAHparams, algorithm_registry)
from composer.core.algorithm import Algorithm

default_required_fields = {
    AlibiHparams: {
        'position_embedding_attribute': 'module.transformer.wpe',
        'attention_module_name': 'transformers.models.gpt2.modeling_gpt2.GPT2Attention',
        'attr_to_replace': '_attn',
        'alibi_attention': 'composer.algorithms.alibi._gpt2_alibi._attn',
        'mask_replacement_function': 'composer.algorithms.alibi._gpt2_alibi.enlarge_mask',
    },
    BlurPoolHparams: {
        'replace_convs': True,
        'replace_maxpools': True,
        'blur_first': True
    },
    LabelSmoothingHparams: {
        'smoothing': 0.1
    },
    LayerFreezingHparams: {
        'freeze_start': 0.5,
        'freeze_level': 1.0
    },
    ChannelsLastHparams: {},
    ColOutHparams: {
        "p_row": 0.15,
        "p_col": 0.15,
        "batch": True,
    },
    FactorizeHparams: {
        "min_channels": 16,
        "latent_channels": 0.5,
    },
    SeqLengthWarmupHparams: {
        "duration": 0.30,
        "min_seq_length": 8,
        "max_seq_length": 1024,
        "step_size": 8,
        "truncate": True,
    },
    CutMixHparams: {
        'alpha': 1.0,
        'num_classes': 1000
    },
    CutOutHparams: {
        'num_holes': 1,
        'length': 0.5
    },
    MixUpHparams: {
        'alpha': 0.2,
    },
    GhostBatchNormHparams: {
        'ghost_batch_size': 32
    },
    ScaleScheduleHparams: {
        'ratio': 0.5
    },
    NoOpModelHparams: {},
    SqueezeExciteHparams: {
        'latent_channels': 64,
        'min_channels': 128
    },
    StochasticDepthHparams: {
        'stochastic_method': 'block',
        'target_layer_name': 'ResNetBottleneck',
        'drop_rate': 0.2,
        'drop_distribution': 'linear',
        'drop_warmup': "0.0dur",
        'use_same_gpu_seed': False
    },
    ProgressiveResizingHparams: {
        'mode': 'resize',
        'initial_scale': 0.5,
        'finetune_fraction': 0.2,
        'resize_targets': False,
    },
    RandAugmentHparams: {
        'severity': 9,
        'depth': 2,
        'augmentation_set': 'all'
    },
    SWAHparams: {
        'swa_start': "0.7dur",
        'swa_end': "0.97dur",
        'update_interval': "1ep",
        'schedule_swa_lr': False,
        'anneal_strategy': 'cos',
        'anneal_steps': 10,
        'swa_lr': None
    },
    AugMixHparams: {
        'severity': 3,
        'depth': 3,
        'width': 3,
        'alpha': 1.0,
        'augmentation_set': 'all'
    },
    SAMHparams: {},
    SelectiveBackpropHparams: {
        'start': 0.5,
        'end': 0.9,
        'keep': 0.5,
        'scale_factor': 0.5,
        'interrupt': 2
    },
}


@pytest.fixture
def registry():
    return algorithm_registry.get_algorithm_registry()


@pytest.mark.parametrize("name", algorithm_registry.list_algorithms())
def test_algorithm_registry(name, registry):
    # create the hparams object
    hparams_class = registry[name]
    hparams = hparams_class(**default_required_fields[hparams_class])
    assert isinstance(hparams, AlgorithmHparams)
    assert dataclasses.is_dataclass(hparams)

    algorithm = algorithm_registry.get_algorithm(hparams)
    assert isinstance(algorithm, Algorithm)
