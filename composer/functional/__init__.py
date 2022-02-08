# Copyright 2021 MosaicML. All Rights Reserved.

"""Functional API for applying algorithms.

.. code-block:: python

    from composer import functional as cf
    from torchvision import models

    model = models.resnet(model_name='resnet50')

    # replace some layers with blurpool
    cf.apply_blurpool(model)
    # replace some layers with squeeze-excite
    cf.apply_se(model, latent_channels=64, min_channels=128)
"""
from composer.algorithms.alibi.alibi import apply_alibi
from composer.algorithms.augmix import augmix_image
from composer.algorithms.blurpool import apply_blurpool
from composer.algorithms.channels_last.channels_last import apply_channels_last
from composer.algorithms.colout.colout import colout_batch, colout_image
from composer.algorithms.cutmix.cutmix import cutmix_batch
from composer.algorithms.cutout.cutout import apply_cutout, cutout_batch
from composer.algorithms.ghost_batchnorm.ghost_batchnorm import apply_ghost_batchnorm
from composer.algorithms.label_smoothing import smooth_labels
from composer.algorithms.layer_freezing import freeze_layers
from composer.algorithms.mixup import mixup_batch
from composer.algorithms.mixup.mixup import gen_interpolation_lambda
from composer.algorithms.progressive_resizing import resize_inputs
from composer.algorithms.randaugment import randaugment
from composer.algorithms.scale_schedule.scale_schedule import scale_scheduler
from composer.algorithms.selective_backprop.selective_backprop import do_selective_backprop, selective_backprop
from composer.algorithms.seq_length_warmup.seq_length_warmup import apply_seq_length_warmup
from composer.algorithms.squeeze_excite import apply_se
from composer.algorithms.stochastic_depth.stochastic_depth import apply_stochastic_depth

# All must be manually defined so sphinx automodule will work properly
__all__ = [
    "apply_alibi",
    "augmix_image",
    "apply_blurpool",
    "apply_channels_last",
    "colout_batch",
    "colout_image",
    "cutmix_batch",
    "cutout_batch",
    "apply_cutout",
    "apply_ghost_batchnorm",
    "smooth_labels",
    "freeze_layers",
    "mixup_batch",
    "gen_interpolation_lambda",
    "resize_inputs",
    "randaugment",
    "scale_scheduler",
    "do_selective_backprop",
    "selective_backprop",
    "apply_seq_length_warmup",
    "apply_se",
    "apply_stochastic_depth",
]
