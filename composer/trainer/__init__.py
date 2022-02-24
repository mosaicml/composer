# Copyright 2021 MosaicML. All Rights Reserved.

"""Train models!

The trainer supports models with :class:`ComposerModel` instances.
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
                      max_duration="50ep",
                      train_batch_size=128,
                      eval_batch_size=128,
                      checkpoint_filepath="checkpoints/first_checkpoint.pt")
    trainer.fit()


.. code-block:: python

    from composer.trainer import TrainerHparamms

    # Create a trainer from hparams and train train the model
    trainer = hparams.initialize_object()
    trainer.fit()
"""
from composer.trainer import devices as devices
from composer.trainer.trainer import Trainer as Trainer
from composer.trainer.trainer_hparams import TrainerHparams as TrainerHparams

load = TrainerHparams.load
