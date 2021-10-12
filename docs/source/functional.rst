composer.functional
===================

Algorithms can be used directly through our functions-based API.

.. code-block:: python

    from composer import functional as CF
    from torchvision import models

    model = models.resnet50()

    # replace some layers with blurpool or squeeze-excite layers
    CF.apply_blurpool(model)
    CF.apply_se(model)


.. currentmodule:: composer.algorithms

.. autosummary::
    :toctree: generated
    :nosignatures:

    functional.augment_and_mix
    functional.apply_blurpool
    functional.colout
    functional.cutout
    functional.smooth_labels
    functional.freeze_layers
    functional.mixup_batch
    functional.resize_inputs
    functional.randaugment
    functional.scale_scheduler
    functional.selective_backprop
    functional.apply_se
    functional.apply_curriculum
