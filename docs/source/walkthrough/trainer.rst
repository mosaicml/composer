composer.trainer
================

.. currentmodule:: composer

:class:`Trainer` is used to train models with :class:`Algorithm` instances.
The :class:`Trainer` is highly customizable and can support a wide variety of workloads.


Examples
--------

.. code-block:: python

    # Setup dependencies
    from composer.datasets import MNISTDatasetHparams
    from composer.models.mnist import MnistClassifierHparams
    model = MnistClassifierHparams(num_classes=10).initialize_objeect()
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


.. code-block:: python

    # Create a trainer that will checkpoint every epoch
    # and train the model
    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      eval_dataloader=eval_dataloader,
                      max_duration="50ep",
                      train_batch_size=128,
                      eval_batch_size=128,
                      checkpoint_interval_unit="ep",
                      checkpoint_folder="checkpoints",
                      checkpoint_interval=1)
    trainer.fit()


.. code-block:: python

    # Load a trainer from the saved checkpoint and resume training
    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      eval_dataloader=eval_dataloader,
                      max_epochs=50,
                      train_batch_size=128,
                      eval_batch_size=128,
                      checkpoint_filepath="checkpoints/first_checkpoint.pt")
    trainer.fit()


.. code-block:: python

    from composer.trainer import TrainerHparamms

    # Create a trainer from hparams and train train the model
    trainer = hparams.initialize_object()
    trainer.fit()


Trainer Hparams
---------------

:class:`Trainer` can be constructed via either it's ``__init__`` (see below)
or
`TrainerHparams <https://github.com/mosaicml/composer/blob/main/composer/trainer/trainer_hparams.py>`_.

Our `yahp <https://github.com/mosaicml/yahp>`_ based system allows configuring the trainer and algorithms via either a ``yaml`` file (see `here <https://github.com/mosaicml/composer/blob/main/composer/yamls/models/classify_mnist_cpu.yaml>`_ for an example) or command-line arguments. Below is a table of all the keys that can be used.

For example, the yaml for ``algorithms`` can include:

.. code-block:: yaml

    algorithms:
        - blurpool
        - layer_freezing


You can also provide overrides at command line:


.. code-block:: bash

    python examples/run_composer_trainer.py -f composer/yamls/models/classify_mnist_cpu.yaml --algorithms blurpool layer_freezing --datadir ~/datasets

API Reference
*************

See :mod:`composer.trainer`.
