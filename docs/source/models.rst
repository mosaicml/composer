composer.models
===============

Models provided to :class:`~composer.trainer.Trainer` must use the basic
interface specified by :class:`~composer.models.BaseMosaicModel`.

Additionally, for convience we provide a number of extensions of :class:`~composer.models.BaseMosaicModel`
as detailed below.

.. currentmodule:: composer.models

Base Models
-----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    BaseMosaicModel
    MosaicClassifier
    MosaicTransformer

Image Models
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    CIFAR10_ResNet56
    MNIST_Classifier
    EfficientNetB0
    ResNet18
    ResNet50
    ResNet101
    UNet

Language Models
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    GPT2Model


Metrics and Loss Functions
--------------------------

Evaluation metrics for common tasks are
in `torchmetrics <https://torchmetrics.readthedocs.io/en/latest/references/modules.html>`_
and are directly compatible with :class:`BaseMosaicModel`.
Additionally, we provide implementations of the following metrics and loss functions.

.. autosummary::
    :toctree: generated
    :nosignatures:

    ~loss.Dice
    ~loss.CrossEntropyLoss
    ~loss.soft_cross_entropy
    ~nlp_metrics.LanguageCrossEntropyLoss
    ~nlp_metrics.Perplexity
