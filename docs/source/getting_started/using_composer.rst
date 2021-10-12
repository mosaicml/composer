Using Composer
==============

We provide several paths to use our library:

* Use our functional API to integrate methods directly into your training loops
* For easy composability, use our MosaicML ``Trainer`` to quickly experiment with different methods.



Functional
~~~~~~~~~~

Almost all algorithms (efficiency methods) are implemented as both a standalone functions for direct access, and as classes for integration into the MosaicML ``Trainer``.

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
    CF.apply_se(model)

As another example, to apply Progressive Resizing, which increases the image size over the course of training:

.. code-block:: python

    from composer import functional as CF

    scale = 0.5
    for (image, label) in your_dataloader:
        CF.resize_inputs(image, label, scale_factor=scale)
        scale += 0.01

For more details, please see :doc:`/functional`


MosaicML Trainer
~~~~~~~~~~~~~~~~

The previous approach is easy to get started and experiment with methods. However, the key to Composer is the ability to quickly configure and compose multiple methods together. For this, use the MosaicML :class:`Trainer`. Our trainer is designed to be minimally more opinionated than other libraries in order to achieve our composition goals.

Our trainer features:

* interface to flexibly add algorithms to the training loop
* engine that manages the ordering of algorithms for composition
* a hyperparameter system based on `yahp`_ (optional, but recommended)

Here are several ways to use ``Trainer``:

1. (Fastest): Directly load the hparams for preconfigured models and algorithms.

   .. code-block:: python

       from composer import algorithms, trainer
       from composer.trainer import Trainer

       hparams = trainer.load("resnet50")  # loads from composer/models/resnet50/resnet50.yaml
       hparams.algorithms = algorithms.load_multiple(["blurpool", "label_smoothing"])

       # edit other properties in the hparams object
       hparams.precision = "fp32"
       hparams.grad_accum = 2

       learner = Trainer.create_from_hparams(hparams)
       learner.fit()

   For a list of properties, see: <add link to docstrings for hparams>

2. (Configurable): Provide a ``yaml`` file, either from our defaults or customized yourself.

    With our `run_mosaic_trainer.py` entrypoint:

   .. code-block::

       git clone git@github.com:mosaicml/composer && cd composer
       python examples/run_mosaic_trainer.py -f composer/models/classify_mnist/classify_mnist_cpu.yaml

   Or, in python,

   .. code-block:: python

        from composer.trainer import trainer_hparams, Trainer

        hparams = trainer_hparams.create('path_to_yaml')
        learner = Trainer.create_from_hparams(hparams)

        learner.fit()

   For more details on `yahp`_, see this primer <TODO: HERE>

3. (Flexible): The :class:`Trainer` can also be initialized directly:

   .. code-block:: python

       from composer.trainer import Trainer
       from composer import models, datasets
       from torchvision import datasets

       train_dataloader_spec = datasets.DataloaderSpec(
           dataset=datasets.MNIST('/datasets/', train=True, download=True),
           drop_last=False,
           shuffle=True,
       )

       eval_dataloader_spec = datasets.DataloaderSpec(
           dataset=datasets.MNIST('/datasets/', train=False, download=True),
           drop_last=False,
           shuffle=False,
       )
       trainer = Trainer(
           model=models.MNIST_Classifier(num_classes=10),
           train_dataloader_spec=train_dataloader_spec,
           eval_dataloader_spec=eval_dataloader_spec,
           max_epochs=3,
           train_batch_size=256,
           eval_batch_size=256,
       )

       trainer.fit()

   For a comprehensive list of training arguments, see: :doc:`/trainer`


.. _yahp: https://github.com/mosaicml/yahp

