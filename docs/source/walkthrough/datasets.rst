composer.datasets
=================

.. currentmodule:: composer.datasets

:class:`DataloaderHparams` contains the :class:`torch.utils.data.dataloader` settings that are common across both training and eval datasets:

* ``num_workers``
* ``prefetch_factor``
* ``persistent_workers``
* ``pin_memory``
* ``timeout``

Each :class:`DatasetHparams` is then responsible for settings such as:

* ``dataset``
* ``drop_last``
* ``shuffle``
* ``collate_fn``

A :class:`DatasetHparams` is responsible for returning a :class:`torch.utils.data.dataloader` or a :class:`DataloaderSpec`.

API Reference
*************

For a list of datasets available in composer, see the :mod:`API Reference <composer.datasets>`.
