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

.. automodule:: composer.algorithms.functional
    :noindex:
    :no-members:
    :autosummary-imported-members:
    :autosummary-members:
