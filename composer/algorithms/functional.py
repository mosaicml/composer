# Copyright 2021 MosaicML. All Rights Reserved.

"""
Functional API

Functional forms of methods are available here via::

    from composer import functional as CF
    my_model = CF.apply_blurpool(my_model)


"""
from composer.algorithms.augmix import augment_and_mix as augment_and_mix
from composer.algorithms.blurpool import apply_blurpool as apply_blurpool
from composer.algorithms.colout.colout import colout as colout
from composer.algorithms.curriculum_learning.curriculum_learning import apply_curriculum as apply_curriculum
from composer.algorithms.cutout.cutout import cutout as cutout
from composer.algorithms.label_smoothing import smooth_labels as smooth_labels
from composer.algorithms.layer_freezing import freeze_layers as freeze_layers
from composer.algorithms.mixup import mixup_batch as mixup_batch
from composer.algorithms.progressive_resizing import resize_inputs as resize_inputs
from composer.algorithms.randaugment import randaugment as randaugment
from composer.algorithms.scale_schedule.scale_schedule import scale_scheduler as scale_scheduler
from composer.algorithms.selective_backprop.selective_backprop import do_selective_backprop as do_selective_backprop
from composer.algorithms.selective_backprop.selective_backprop import selective_backprop as selective_backprop
from composer.algorithms.squeeze_excite import apply_se as apply_se
