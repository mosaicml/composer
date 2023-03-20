|:cd:| DataLoaders
==================

DataLoaders are used to pass in training or evaluation data to the
Composer :class:`.Trainer`. There are three different ways of doing so:

1. Passing PyTorch :class:`torch.utils.data.DataLoader` objects directly.
2. Providing a |DataSpec|, which contains a PyTorch dataloader as well as
   additional configurations, such as on-device transforms.
3. (For validation) Providing :class:`.Evaluator` objects which contain both a
   dataloader and relevant metrics for validation.

We walk through each of these ways in detail and provide usage examples
below.

Passing a PyTorch DataLoader
----------------------------

Composer dataloaders have type :class:`torch.utils.data.DataLoader`
(see `PyTorch`_ documentation) and can be passed directly to the
:class:`.Trainer`.

.. code:: python

    from torch.utils.data import DataLoader
    from composer import Trainer

    train_dataloader = DataLoader(
        training_data,
        batch_size=2048,
        shuffle=True
    )

    trainer = Trainer(..., train_dataloader=train_dataloader, ...)

.. note::

    The ``batch_size`` to the dataloader should be the per-device overall batch size. For example,
    if you were using ``device_train_microbatch_size=1024``, a batch_size of ``2048`` would mean
    that each *microbatch* (one forward/backward pass) would have a batch size of ``1024``.

For performance, we highly recommend:

-  ``num_workers > 0`` : usually set this to the number of CPU cores in
   your machine divided by the number of GPUs.
-  ``pin_memory = true`` : Pinned memory can speed up copying memory
   from a CPU to a GPU. Try to use it everywhere possible because
   the only drawback is the reduced RAM available to the host.
-  ``persistent_workers = true`` : Persisting workers will reduce the
   overhead of creating workers but will use some RAM since these workers
   have some persistent state.

.. note::
    Samplers are used to specify the order of indices in dataloading. When
    using distributed training, it is important to use the Torch
    :class:`~torch.utils.data.distributed.DistributedSampler`.
    so that each process sees a unique shard of the dataset. If the dataset
    is already sharded, then use a :class:`~torch.utils.data.SequentialSampler`
    or :class:`~torch.utils.data.RandomSampler`.

DataSpec
--------

Sometimes, the data configuration requires more than just the dataloader. Some example
additional configurations include:

-  Some transforms should be run on the data after it has been moved
   onto the correct device (e.g. ``GPU``).
-  Custom batch types would need a ``split_batch`` function that tells
   our trainer how to split the batches into microbatches for gradient
   accumulation.
-  Optionally tracking the number of tokens (or samples) seen during
   training so far.
-  Providing the length of a dataset when ``len`` or a similar function
   isn't in the dataloader's interface.

For these and other potential uses cases, the trainer can also accept the
|DataSpec| object with these additional settings. For example,

.. code:: python

    from composer import Trainer
    from composer.core import DataSpec

    data_spec = DataSpec(
        dataloader=my_train_dataloader,
        num_tokens=193820,
        get_num_tokens_in_batch=lambda batch: batch['text'].shape[0]
    )

    trainer = Trainer(train_dataloader=data_spec, ...)

Examples of how |DataSpec| is used for popular datasets can be seen in
our `ImageNet`_ and `ADE20k`_ files. For reference, the |DataSpec| arguments
are shown below.

.. currentmodule:: composer.core

.. autoclass:: DataSpec
    :noindex:

Validation
----------

To track training progress, validation datasets can be provided to the
Composer Trainer through the ``eval_dataloader`` parameter. If there are
multiple datasets to use for validation/evaluation, each
with their own metrics, :class:`.Evaluator` objects can be used to
pass in multiple dataloaders/datasets to the trainer.

For more information, see :doc:`Evaluation</trainer/evaluation>`.


Batch Types
-----------

For custom batch types (not torch.Tensor, List, Tuple, Mapping), implement and provide
the ``split_batch`` function to the trainer using :class:`.DataSpec` above. Here's an
example function or when the batch from the dataloader is a tuple of two tensors:

.. code:: python

    def split_batch(self, batch: Batch, num_microbatches: int) -> List[Batch]:
        x, y = batch
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return list(zip(x.chunk(num_microbatches), y.chunk(num_microbatches)))

Suppose instead the batch had one input image and several target images,
e.g. ``(Tensor, (Tensor, Tensor, Tensor))``. Then the function would be:

.. code:: python

    def split_batch(self, batch: Batch, num_microbatches: int) -> List[Batch]:
        n = num_microbatches

        x, (y1, y2) = batch
        chunked = (x.chunk(n), (y1.chunk(n), y2.chunk(n)))
        return list(zip(*chunked))


.. |DataSpec| replace:: :class:`~composer.core.DataSpec`
.. _ImageNet: https://github.com/mosaicml/composer/blob/dev/composer/datasets/imagenet.py
.. _ADE20k: https://github.com/mosaicml/composer/blob/dev/composer/datasets/ade20k.py
.. _pytorch: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
