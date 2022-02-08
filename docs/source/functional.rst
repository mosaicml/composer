composer.functional
===================

Algorithms can be used directly through our functions-based API.

.. code-block:: python

    from composer import functional as cf
    from torchvision import models

    model = models.resnet(model_name='resnet50')

    # replace some layers with blurpool
    cf.apply_blurpool(model)
    # replace some layers with squeeze-excite
    cf.apply_se(model, latent_channels=64, min_channels=128)


.. currentmodule:: composer

.. autosummary::
    :toctree: generated
    :nosignatures:

    functional.apply_alibi
    functional.apply_blurpool
    functional.apply_channels_last
    functional.apply_factorization
    functional.apply_ghost_batchnorm
    functional.apply_squeeze_excite
    functional.apply_stochastic_depth
    functional.augmix_image
    functional.colout_batch
    functional.colout_image
    functional.cutmix_batch
    functional.cutout_batch
    functional.freeze_layers
    functional.gen_mixup_interpolation_lambda
    functional.mixup_batch
    functional.randaugment_image
    functional.resize_batch
    functional.scale_scheduler
    functional.selective_backprop
    functional.set_batch_sequence_length
    functional.smooth_labels
