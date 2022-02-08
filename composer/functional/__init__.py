# Copyright 2021 MosaicML. All Rights Reserved.

"""Functional API.

Functional forms of methods are available here via::

    from composer import functional as cf
    my_model = cf.apply_blurpool(my_model)
"""
from composer.algorithms.alibi.alibi import apply_alibi as apply_alibi
from composer.algorithms.augmix import augmix_image as augmix_image
from composer.algorithms.blurpool import apply_blurpool as apply_blurpool
from composer.algorithms.channels_last import apply_channels_last as apply_channels_last
from composer.algorithms.colout import colout_batch as colout_batch
from composer.algorithms.colout import colout_image as colout_image
from composer.algorithms.cutmix import cutmix_batch as cutmix_batch
from composer.algorithms.cutout import cutout_batch as cutout_batch
from composer.algorithms.factorize import apply_factorization as apply_factorization
from composer.algorithms.ghost_batchnorm.ghost_batchnorm import apply_ghost_batchnorm as apply_ghost_batchnorm
from composer.algorithms.label_smoothing import smooth_labels as smooth_labels
from composer.algorithms.layer_freezing import freeze_layers as freeze_layers
from composer.algorithms.mixup import gen_mixup_interpolation_lambda as gen_mixup_interpolation_lambda
from composer.algorithms.mixup import mixup_batch as mixup_batch
from composer.algorithms.progressive_resizing import resize_batch as resize_batch
from composer.algorithms.randaugment import randaugment_image as randaugment_image
from composer.algorithms.scale_schedule import scale_scheduler as scale_scheduler
from composer.algorithms.selective_backprop import selective_backprop as selective_backprop
from composer.algorithms.seq_length_warmup import set_batch_sequence_length as set_batch_sequence_length
from composer.algorithms.squeeze_excite import apply_squeeze_excite as apply_squeeze_excite
from composer.algorithms.stochastic_depth import apply_stochastic_depth as apply_stochastic_depth
