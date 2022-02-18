Installation
============

MosaicML ``Composer`` requires Python 3.7+ and Pytorch 1.9+. It can be installed via ``pip``:

.. code-block:: console

    pip install mosaicml

To include non-core dependencies that are required by some algorithms, callbacks, datasets, and models,
the following installation targets are available:

* ``pip install mosaicml[deepspeed]``: Installs Composer with support for :mod:`deepspeed`.
* ``pip install mosaicml[nlp]``: Installs Composer with support for NLP models and algorithms
* ``pip install mosaicml[unet]``: Installs Composer with support for :doc:`Unet </model_cards/unet>`
* ``pip install mosaicml[timm]``: Installs Composer with support for :mod:`timm`
* ``pip install mosaicml[wandb]``: Installs Composer with support for :mod:`wandb`.
* ``pip install mosaicml[dev]``: Installs development dependencies, which are required for running tests
  and building documentation.
* ``pip install mosaicml[all]``: Install all optional dependencies

For a developer install, clone directly:

.. code-block:: console

    git clone https://github.com/mosaicml/composer.git
    cd composer
    pip install -e .[all]


.. note::

    For performance in image-based operations, we **highly** recommend installing 
    Pillow-SIMD<https://github.com/uploadcare/pillow-simd>`_ To install, vanilla pillow must first be uninstalled.

    .. code-block:: console

        pip uninstall pillow && pip install pillow-simd

    Pillow-SIMD is not supported for Apple M1 Macs.


Docker
~~~~~~

To access our docker, either pull the latest image from our Docker repository with:

.. code-block::

    docker pull mosaicml/composer:latest

or build our ``Dockerfile``:

.. code-block::

    git clone https://github.com/mosaicml/composer.git
    cd composer/docker && make build

Our dockerfile has Ubuntu 18.04, Python 3.8.0, PyTorch 1.9.0, and CUDA 11.1.1, and has been tested to work with
GPU-based instances on AWS, GCP, and Azure. ``Pillow-SIMD`` is installed by default in our docker image.

Please see the ``README`` in the docker area for additional details.


Verification
~~~~~~~~~~~~

Test ``Composer`` was installed properly by opening a ``python`` prompt, and run:

.. code-block:: python

    import logging
    from composer import functional as CF
    import torchvision.models as models

    logging.basicConfig(level=logging.INFO)
    model = models.resnet(model_name='resnet50')

    CF.apply_blurpool(model)

This creates a ResNet50 model and replaces several pooling and convolution layers with BlurPool variants
(`Zhang et al, 2019<https://arxiv.org/abs/1904.11486>`_). The method should log:

.. code-block:: none

    Applied BlurPool to model ResNet Model now has 1 BlurMaxPool2d and 6 BlurConv2D layers.

Next, train a small classifier on MNIST with the label smoothing algorithm:

.. code-block::

    git clone https://github.com/mosaicml/composer.git
    cd composer
    pip install -e .
    python examples/run_composer_trainer.py -f composer/yamls/models/classify_mnist_cpu.yaml --datadir ~/datasets/ --algorithms label_smoothing --alpha 0.1
