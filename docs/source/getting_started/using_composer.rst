Using Composer
==============

We provide several paths to use our library:

* Use our :ref:`using_composer_functional` to integrate methods directly into your training loops
* For easy composability, use our :ref:`using_composer_trainer` to quickly experiment with different methods.

.. _using_composer_functional:

Functional API
~~~~~~~~~~~~~~

Almost all algorithms (efficiency methods) are implemented as both standalone functions for direct access, and as classes for integration into the :ref:`using_composer_trainer`.

For example, to use some of our model surgery-based methods, apply those functions after model creation and before the optimizers are created. First, to understand what is being modified, enable logging:

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.INFO)

Then, we apply BlurPool and SqueezeExcite methods to replace eligible convolution layers:

.. code-block:: python

    from composer import functional as CF
    import torchvision
    model = torchvision.models.resnet50()

    # replace some layers with blurpool or squeeze-excite layers
    CF.apply_blurpool(model)
    CF.apply_se(model, latent_channels=64, min_channels=128)

As another example, to apply Progressive Resizing, which increases the image size over the course of training:

.. code-block:: python

    from composer import functional as CF

    scale = 0.5
    for (image, label) in your_dataloader:
        CF.resize_inputs(image, label, scale_factor=scale)
        scale += 0.01

For more details, please see :mod:`composer.functional`.

.. _using_composer_trainer:

Composer Trainer
~~~~~~~~~~~~~~~~

The previous approach is easy to get started and experiment with methods. However, the key to Composer is the ability to quickly configure and compose multiple methods together. For this, use the Composer Trainer. The trainer is designed to be minimally more opinionated than other libraries in order to achieve our composition goals.

Our trainer features:

* interface to flexibly add algorithms to the training loop
* engine that manages the ordering of algorithms for composition
* a hyperparameter system based on `yahp`_ (optional, but recommended)

Here are several ways to use the trainer:

1. (Fastest): Directly load the hparams for preconfigured models and algorithms.

   .. code-block:: python

       from composer import algorithms, trainer, Trainer
       from composer.core.types import Precision

       hparams = trainer.load("classify_mnist_cpu")  # loads from composer/yamls/models/classify_mnist_cpu.yaml
       hparams.algorithms = algorithms.load_multiple("blurpool", "label_smoothing")

       # edit other properties in the hparams object
       hparams.precision = Precision.FP32
       hparams.grad_accum = 2
       hparams.datadir = "~/datasets"

       trainer = hparams.initialize_object()
       trainer.fit()

   For a list of properties, see: :class:`~composer.trainer.trainer.Trainer`

2. (Configurable): Provide a ``yaml`` file, either from our defaults or customized yourself.

    With our ``run_composer_trainer.py`` entrypoint:

   .. code-block::

       git clone https://github.com/mosaicml/composer.git
       cd composer && pip install -e .
       python examples/run_composer_trainer.py -f composer/yamls/models/classify_mnist_cpu.yaml  --datadir ~/datasets

   Or, in Python,

   .. code-block:: python

        from composer.trainer import TrainerHparams, Trainer

        hparams = TrainerHparams.create('composer/yamls/models/classify_mnist_cpu.yaml')
        hparams.datadir = "~/datasets"
        trainer = hparams.initialize_object()

        trainer.fit()

  For more details on `yahp`_, see the `documentation<https://mosaicml-yahp.readthedocs-hosted.com/en/stable/>`_.

3. (Flexible): The :class:`~composer.trainer.trainer.Trainer` can also be initialized directly:

   .. code-block:: python

        from composer import Trainer
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        train_dataloader = DataLoader(
            datasets.MNIST('~/datasets/', train=True, transform=transforms.ToTensor(), download=True),
            drop_last=True,
            shuffle=True,
            batch_size=256,
        )

        eval_dataloader = DataLoader(
            datasets.MNIST('~/datasets/', train=True, transform=transforms.ToTensor(), download=True),
            drop_last=False,
            shuffle=False,
            batch_size=256,
        )

        trainer = Trainer(
            model=models.MNIST_Classifier(num_classes=10),
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            max_epochs=3,
        )

        trainer.fit()

   For a comprehensive list of training arguments, see :class:`~composer.trainer.trainer.Trainer`.


.. _yahp: https://github.com/mosaicml/yahp
