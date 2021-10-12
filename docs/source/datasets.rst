composer.datasets
=================

.. currentmodule:: composer.datasets

:class:`DataloaderHparams` contain the :class:`torch.utils.data.dataloader` settings that are common across both training and eval datasets:

* ``num_workers``
* ``prefetch_factor``
* ``persistent_workers``
* ``pin_memory``
* ``timeout``

Each :class:`DatasetHparams` is then reponsible for returning a ``DataloaderSpec``, which is a ``namedtuple`` of dataset-specific settings such as:
* ``dataset``
* ``drop_last``
* ``shuffle``
* ``collate_fn``

Ths indirection (instead of directly creating the ``dataloader`` at the start) is needed because for multi-GPU training, dataloaders require the global rank to initialize their :class:`torch.utils.data.distributed.DistributedSampler`.

As a result, our trainer uses the ``DataloaderSpec`` and ``DataloaderHparams`` to create the dataloaders after DistributdDataParallel has forked the processes.


Base objects
------------

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


