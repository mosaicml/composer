# Copyright 2021 MosaicML. All Rights Reserved.

"""Functional API for applying algorithms.

.. code-block:: python

    from composer import functional as cf
    from torchvision import models

    model = models.resnet50()

    # replace some layers with blurpool
    cf.apply_blurpool(model)
    # replace some layers with squeeze-excite
    cf.apply_squeeze_excite(model, latent_channels=64, min_channels=128)
"""
from composer.algorithms.alibi.alibi import apply_alibi as apply_alibi
from composer.algorithms.augmix import augmix_image as augmix_image
from composer.algorithms.blurpool import apply_blurpool as apply_blurpool
from composer.algorithms.channels_last import apply_channels_last as apply_channels_last
from composer.algorithms.colout import colout_batch as colout_batch
from composer.algorithms.cutmix import cutmix_batch as cutmix_batch
from composer.algorithms.cutout import cutout_batch as cutout_batch
from composer.algorithms.factorize import apply_factorization as apply_factorization
from composer.algorithms.ghost_batchnorm.ghost_batchnorm import apply_ghost_batchnorm as apply_ghost_batchnorm
from composer.algorithms.label_smoothing import smooth_labels as smooth_labels
from composer.algorithms.layer_freezing import freeze_layers as freeze_layers
from composer.algorithms.mixup import mixup_batch as mixup_batch
from composer.algorithms.progressive_resizing import resize_batch as resize_batch
from composer.algorithms.randaugment import randaugment_image as randaugment_image
from composer.algorithms.selective_backprop import select_using_loss as select_using_loss
from composer.algorithms.selective_backprop import should_selective_backprop as should_selective_backprop
from composer.algorithms.seq_length_warmup import set_batch_sequence_length as set_batch_sequence_length
from composer.algorithms.squeeze_excite import apply_squeeze_excite as apply_squeeze_excite
from composer.algorithms.stochastic_depth import apply_stochastic_depth as apply_stochastic_depth

# All must be manually defined so sphinx automodule will work properly
__all__ = [
    "apply_alibi",
    "augmix_image",
    "apply_blurpool",
    "apply_channels_last",
    "colout_batch",
    "cutmix_batch",
    "cutout_batch",
    "apply_factorization",
    "apply_ghost_batchnorm",
    "smooth_labels",
    "freeze_layers",
    "mixup_batch",
    "resize_batch",
    "randaugment_image",
    "should_selective_backprop",
    "select_using_loss",
    "set_batch_sequence_length",
    "apply_squeeze_excite",
    "apply_stochastic_depth",
]
