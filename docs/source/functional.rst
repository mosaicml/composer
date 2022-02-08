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

    functional.augmix_image
    functional.apply_blurpool
    functional.apply_alibi
    functional.colout_batch
    functional.colout_image
    functional.cutout_batch
    functional.smooth_labels
    functional.freeze_layers
    functional.apply_ghost_batchnorm
    functional.mixup_batch
    functional.resize_batch
    functional.randaugment_image
    functional.scale_scheduler
    functional.selective_backprop
    functional.apply_squeeze_excite
    functional.apply_factorization
    functional.set_batch_sequence_length
    functional.apply_stochastic_depth
