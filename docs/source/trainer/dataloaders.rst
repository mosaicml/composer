Dataloaders
===========

Dataloaders are used to pass in training or evaluation data to the
Composer :class:`.Trainer`. There are three different ways of doing so:

1. Passing PyTorch :class:`torch.utils.data.DataLoader` objects directly.
2. Providing a :class:`.DataSpec`, which contains a pytorch dataloader, as well as
   additional configurations, such as on-device transforms.
3. (For validation) Providing :class:`.Evaluator` objects which contain both a
   dataloader but also relevant metrics for validation.

We walk through each of these ways in detail and provide usage examples
below.

Passing a PyTorch Dataloader
----------------------------

Composer dataloaders have type :class:`torch.utils.data.DataLoader`
(see `pytorch`_` documentation), and can be passed directly to the
:class:`.Trainer`.

.. code:: python

   from torch.utils.data import DataLoader
   from composer import Trainer

   train_dataloader = DataLoader(
       training_data,
     batch_size=2048,
     shuffle=True)

   trainer = Trainer(.., train_dataloader=train_dataloader, ..)

.. note::

    The ``batch_size`` to the dataloader should be the per-device overall
    batch size. For example, if you are using ``grad_accum=2``, a batch_size
    of ``2048`` means that each *microbatch*\ ( one forward/backward pass) would
    have a batch size of ``1024``.

For performance, we highly recommend:

-  ``num_workers > 0`` : usually set this to the number of CPU cores in
   your machine divided by the number of GPUs.
-  ``pin_memory = true`` : Pinned memory can speed up copying memory
   from a CPU to a GPU. Try to set use it everywhere possible because
   the only drawback is the reduced RAM available to the host.
-  ``persistent_workers = true`` : Persisting workers across will reduce the
   overhead of creating workers, but will use some RAM since those workers
   have some persistent state.

.. note::
    Samplers are used to specify order of indices in dataloading. When
    using distributed training, it is important to use the torch
    :class:`~torch.utils.data.distributed.DistributedSampler`.
    so that each process sees a unique shard of the dataset. If the dataset
    is already sharded, then use a :class:`~torch.utils.data.SequentialSampler`
    or :class:`~torch.utils.data.RandomSampler`.

DataSpec
--------

Sometimes users may need to provide more configurations than just the
PyTorch DataLoader:

-  Some transforms that should be run on the device (e.g. ``GPU``) after
   the data has been moved.
-  Custom batch types would need a ``split_batch`` function that tells
   our trainer how to split the batches into microbatches for gradient
   accumulation.
-  To optionally track the number of tokens (or samples) seen during
   training so far.
-  Provide the length of a dataset when ``len`` or similar function
   isn't in the dataloader's interface.

For these and other potential uses cases, the trainer can also accept
:class:`.DataSpec` object with these additional configurations.

.. code:: python

   from composer import Trainer
   from composer.core import DataSpec

   data_spec = DataSpec(
       dataloader=my_train_dataloader,
       num_tokens=193820,
       get_num_tokens_in_batch=lambda batch: batch['data'].shape[0]
   )

   trainer = Trainer(train_dataloader=data_spec, ...)

Examples of how :class:`.DataSpec` is used for popular datasets can be seen in
our ImageNet and ADE20k files.

.. currentmodule:: composer.core

.. autoclass:: DataSpec


Validation
----------

To track training progress, validation datasets can be provided to the
Composer Trainer through the ``eval_dataloader`` parameter. The trainer
will compute evaluation metrics on the evaluation dataset at a frequency
specified by the the :class:`.Trainer` parameters ``validate_every_n_batches``
and ``validate_every_n_epochs`` .

Example
~~~~~~~

.. code:: python

   from composer import Trainer

   trainer = Trainer(
           ...,
           eval_dataloader=my_eval_dataloader,
           validate_every_n_batches = 100, # Default is -1 to not evaluate on batchwise frequency
       validate_every_n_epochs = 1 # Default is 1
   )

Multiple Validation Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If there are multiple datasets to use for validation/evaluation, each
with their own evaluation metrics, :class:`.Evaluator` objects can be used to
pass in multiple dataloaders/datasets to the trainer. For more
information, see Evaluation.
