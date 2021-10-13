Installation
============

MosaicML ``Composer`` can be installed via ``pip``:

.. code-block::

    pip install mosaicml

Additional dependencies can be installed by specifying ``mosaicml[tag]``. The following tags are available:

- ``nlp``: installs huggingface ``transformers`` and ``datasets``
- ``dev``: packages needed for testing, linting, and docs.
- ``wandb``: enables the weights & biases logger
- ``unet``: enables the U-Net model and BRATS dataset
- ``all``: installs all of the above.

For a developer install, clone directly:

.. code-block::

    git clone git@github.com/mosaicml/composer.git
    cd composer && pip install -e .


.. note::

    For performance in image-based operations, we **highly** recommend installing `Pillow-SIMD <https://github.com/uploadcare/pillow-simd>`_. To install, vanilla pillow must first be uninstalled.

    .. code-block::

        pip uninstall pillow && pip install pillow-simd

    Pillow-SIMD is not supported for Apple M1 Macs.

``Composer`` has been tested with Ubuntu 20.04, PyTorch 1.8.1, and Python 3.8.

Docker
~~~~~~

To access our docker, either pull the latest image from our Docker repository with:

.. code-block::

    docker pull mosaicml/composer:latest

or build our ``Dockerfile``:

.. code-block::

    git clone git@github.com:mosaicml/composer.git
    cd composer/docker && make build

Our dockerfile has Ubuntu 18.04, Python 3.8.0, PyTorch 1.9.0, and CUDA 11.1.1, and has been tested to work with GPU-based instances on AWS, GCP, and Azure. ``Pillow-SIMD`` is installed by default in our docker image.

Please see the ``README`` in the docker area for additional details.


Verification
~~~~~~~~~~~~

Test ``Composer`` was installed properly by opening a ``python`` prompt, and running:

.. code-block:: python

    import logging
    from composer import functional as CF
    import torchvision.models as models

    logging.basicConfig(level=logging.INFO)
    model = models.resnet50()

    CF.apply_blurpool(model)

This creates a ResNet50 model and replaces several pooling and convolution layers with BlurPool variants (`Zhang et al, 2019 <https://arxiv.org/abs/1904.11486>`_. The method should log:

.. code-block:: none

    Applied BlurPool to model ResNet Model now has 1 BlurMaxPool2d and 6 BlurConv2D layers.

Next, train a small classifier on MNIST with the label smoothing algorithm:

.. code-block::

    git clone git@github.com:mosaicml/composer.git
    cd composer
    python examples/run_mosaic_trainer.py -f composer/models/classify_mnist/hparams_cpu.yaml --datadir /datasets/ --algorithms label_smoothing --alpha 0.1
