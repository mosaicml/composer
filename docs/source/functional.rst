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
    cf.apply_squeeze_excite(model, latent_channels=64, min_channels=128)


.. currentmodule:: composer.algorithms

.. autosummary::
    :toctree: generated
    :nosignatures:

    functional.augmix_image
    functional.apply_blurpool
    functional.apply_alibi
    functional.colout
    functional.cutout
    functional.smooth_labels
    functional.freeze_layers
    functional.apply_ghost_batchnorm
    functional.mixup_batch
    functional.resize_inputs
    functional.randaugment
    functional.scale_scheduler
    functional.selective_backprop
    functional.apply_se
    functional.apply_seq_length_warmup
    functional.apply_stochastic_depth
