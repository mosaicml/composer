|:floppy_disk:| Installation
============================

``Composer`` is available with Pip:

.. code-block::

    pip install mosaicml

``Composer`` is also available via Anaconda:

.. code-block::

    conda install -c mosaicml composer

To include non-core dependencies that are required by some algorithms, callbacks, datasets, and models,
the following installation targets are available:

* ``pip install mosaicml[dev]``: Installs development dependencies, which are required for running tests
  and building documentation.
* ``pip install mosaicml[deepspeed]``: Installs Composer with support for :mod:`deepspeed`.
* ``pip install mosaicml[nlp]``: Installs Composer with support for NLP models and algorithms
* ``pip install mosaicml[unet]``: Installs Composer with support for :doc:`Unet </model_cards/unet>`
* ``pip install mosaicml[timm]``: Installs Composer with support for :mod:`timm`
* ``pip install mosaicml[wandb]``: Installs Composer with support for :mod:`wandb`.
* ``pip install mosaicml[all]``: Install all optional dependencies

For a developer install, clone directly:

.. code-block::

    git clone https://github.com/mosaicml/composer.git
    cd composer
    pip install -e .[all]


.. note::

    For performance in image-based operations, we **highly** recommend installing
    `Pillow-SIMD <https://github.com/uploadcare/pillow-simd>`_\.  To install, vanilla pillow must first be uninstalled.

    .. code-block::

        pip uninstall pillow && pip install pillow-simd

    Pillow-SIMD is not supported for Apple M1 Macs.


.. Docker Section H2 

.. include:: ../../../docker/README.md
   :parser: myst_parser.sphinx_

Verification
~~~~~~~~~~~~

Test ``Composer`` was installed properly by opening a ``python`` prompt, and run:

.. testcode::

    import logging
    from composer import functional as CF
    import torchvision.models as models

    logging.basicConfig(level=logging.INFO)
    model = models.resnet50()

    CF.apply_blurpool(model)

This creates a ResNet50 model and replaces several pooling and convolution layers with
BlurPool variants (`Zhang et al, 2019 <https://arxiv.org/abs/1904.11486>`_). The method should log:

.. code-block:: none

    Applied BlurPool to model ResNet Model now has 1 BlurMaxPool2d and 6 BlurConv2D layers.

Next, train a small classifier on MNIST with the label smoothing algorithm:

.. code-block:: python

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    from composer import Trainer
    from composer.models import MNIST_Classifier
    from composer.algorithms import LabelSmoothing

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=128)

    trainer = Trainer(
        model=MNIST_Classifier(num_classes=10),
        train_dataloader=train_dataloader,
        max_duration="2ep",
        algorithms=[LabelSmoothing(alpha=0.1)]
    )
    trainer.fit()
