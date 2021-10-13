composer.datasets
=================

.. currentmodule:: composer.datasets

:class:`DataloaderHparams` contains the :class:`torch.utils.data.dataloader` settings that are common across both training and eval datasets:

* ``num_workers``
* ``prefetch_factor``
* ``persistent_workers``
* ``pin_memory``
* ``timeout``

Each :class:`DatasetHparams` is then responsible for returning a :class:`DataloaderSpec`, which is a ``NamedTuple`` of dataset-specific settings such as:

* ``dataset``
* ``drop_last``
* ``shuffle``
* ``collate_fn``

This indirection (instead of directly creating the ``dataloader`` at the start) is needed because for multi-GPU training, dataloaders require the global rank to initialize their :class:`torch.utils.data.distributed.DistributedSampler`.

As a result, our trainer uses the :class:`DataloaderSpec` and :class:`DataloaderHparams` to create the dataloaders after DDP has forked the processes.


Base Classes and Hyperparameters
--------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    DataloaderHparams
    DataloaderSpec
    DatasetHparams

Datasets
--------

.. autosummary::
    :toctree: generated
    :nosignatures:

    MNISTDatasetHparams
    CIFAR10DatasetHparams
    ImagenetDatasetHparams
    LMDatasetHparams
    SyntheticDatasetHparams
    BratsDatasetHparams


