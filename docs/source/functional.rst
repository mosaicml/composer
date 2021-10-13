composer.functional
===================

Algorithms can be used directly through our functions-based API.

.. code-block:: python

    from composer import functional as CF
    from torchvision import models

    model = models.resnet50()

    # replace some layers with blurpool
    CF.apply_blurpool(model)
    # replace some layers with squeeze-excite
    CF.apply_se(model, latent_channels=64, min_channels=128)


.. currentmodule:: composer.algorithms

.. autosummary::
    :toctree: generated
    :nosignatures:

    functional.augment_and_mix
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
