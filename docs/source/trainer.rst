composer.trainer
================

.. currentmodule:: composer.trainer

:class:`Trainer` is used to train models with :class:`~composer.Algorithm` instances.
The :class:`Trainer` is highly customizable and can support a wide variety of workloads.


Examples
--------

.. code-block:: python

    # Setup dependencies
    from composer.datasets import MNISTDatasetHparams
    from composer.models.mnist import MnistClassifierHparams
    model = MnistClassifierHparams(num_classes=10).initialize_objeect()
    train_dataloader_spec = MNISTDatasetHparams(is_train=True,
                                                datadir="./mymnist",
                                                download=True).initialize_object()
    train_dataloader_spec = MNISTDatasetHparams(is_train=False,
                                                datadir="./mymnist",
                                                download=True).initialize_object()


.. code-block:: python

    # Create a trainer that will checkpoint every epoch
    # and train the model
    trainer = Trainer(model=model,
                      train_dataloader_spec=train_dataloader_spec,
                      eval_dataloader_spec=eval_dataloader_spec,
                      max_epochs=50,
                      train_batch_size=128,
                      eval_batch_size=128,
                      checkpoint_interval_unit="ep",
                      checkpoint_folder="checkpoints",
                      checkpoint_interval=1)
    trainer.fit()


.. code-block:: python

    # Load a trainer from the saved checkpoint and resume training
    trainer = Trainer(model=model,
                      train_dataloader_spec=train_dataloader_spec,
                      eval_dataloader_spec=eval_dataloader_spec,
                      max_epochs=50,
                      train_batch_size=128,
                      eval_batch_size=128,
                      checkpoint_filepath="checkpoints/first_checkpoint.pt")
    trainer.fit()


.. code-block:: python

    from composer.trainer import TrainerHparamms

    # Create a trainer from hparams and train train the model
    trainer = Trainer.create_from_hparams(hparams=hparams)
    trainer.fit()


API Reference
-------------

.. autoclass:: Trainer
    :members: