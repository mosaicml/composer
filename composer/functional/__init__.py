# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Functional API for applying algorithms in your own training loop.

.. code-block:: python

    from composer import functional as cf
    from torchvision import models

    model = models.resnet50()

    # replace some layers with blurpool
    cf.apply_blurpool(model)
    # replace some layers with squeeze-excite
    cf.apply_squeeze_excite(model, latent_channels=64, min_channels=128)
"""
from composer.algorithms.alibi import apply_alibi
from composer.algorithms.augmix import augmix_image
from composer.algorithms.blurpool import apply_blurpool
from composer.algorithms.channels_last import apply_channels_last
from composer.algorithms.colout import colout_batch
from composer.algorithms.cutmix import cutmix_batch
from composer.algorithms.cutout import cutout_batch
from composer.algorithms.ema import compute_ema
from composer.algorithms.factorize import apply_factorization
from composer.algorithms.fused_layernorm import apply_fused_layernorm
from composer.algorithms.gated_linear_units import apply_gated_linear_units
from composer.algorithms.ghost_batchnorm import apply_ghost_batchnorm
from composer.algorithms.gradient_clipping import apply_gradient_clipping
from composer.algorithms.gyro_dropout import apply_gyro_dropout
from composer.algorithms.label_smoothing import smooth_labels
from composer.algorithms.layer_freezing import freeze_layers
from composer.algorithms.low_precision_layernorm import apply_low_precision_layernorm
from composer.algorithms.mixup import mixup_batch
from composer.algorithms.progressive_resizing import resize_batch
from composer.algorithms.randaugment import randaugment_image
from composer.algorithms.selective_backprop import select_using_loss, should_selective_backprop
from composer.algorithms.seq_length_warmup import set_batch_sequence_length
from composer.algorithms.squeeze_excite import apply_squeeze_excite
from composer.algorithms.stochastic_depth import apply_stochastic_depth
from composer.algorithms.weight_standardization import apply_weight_standardization

# All must be manually defined so sphinx automodule will work properly
__all__ = [
    'apply_alibi',
    'augmix_image',
    'apply_blurpool',
    'apply_channels_last',
    'colout_batch',
    'compute_ema',
    'cutmix_batch',
    'cutout_batch',
    'apply_factorization',
    'apply_fused_layernorm',
    'apply_gated_linear_units',
    'apply_ghost_batchnorm',
    'apply_gradient_clipping',
    'apply_low_precision_layernorm',
    'smooth_labels',
    'freeze_layers',
    'mixup_batch',
    'resize_batch',
    'randaugment_image',
    'should_selective_backprop',
    'select_using_loss',
    'set_batch_sequence_length',
    'apply_squeeze_excite',
    'apply_stochastic_depth',
    'apply_weight_standardization',
    'apply_gyro_dropout',
]
