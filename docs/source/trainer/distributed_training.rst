|:family_man_woman_girl_boy:| Distributed Training
==================================================

Composer supports distributed training on multiple devices, whether it
be multiple GPUs on a single node, or multiple GPUs across multiple
nodes.

Data Parallelism
----------------

Composer distributes work across devices via **data-parallelism-only**.
We choose this in order to provide the most flexibility to algorithms,
which can modify the training loop in complex ways. Data parallelism
greatly simplifies model building and memory management. Every GPU is
performing the same work, so inspecting the rank zero is sufficient to
reason about memory, performance, and other properties.

Within Composer, we have two options for data-parallelism-only
execution: Pytorch DDP and DeepSpeed Zero. We currently default to
Pytorch DDP, though DeepSpeed Zero can provide better performance and
lower memory utilization when configured correctly.

.. note::

    We expect future versions of Composer will default to Zero with auto
    configuration for best performance.

Memory Usage
------------

Composer also supports reducing peak memory usage during training, to
allow **any model to run anywhere:**

-  Inside the Trainer, we split device minibatches into device
   microbatches and train with gradient accumulation, greatly reducing
   activation memory. Use ``grad_accum`` argument to enable this.

.. note::

    The ``batch_size`` passed to your dataloader should be the device
    *mini*\ batch size. We further split this into smaller microbatches with
    gradient accumulation.

-  We currently also support optimizer and gradient sharing via
   Zero stages 1 and 2 respectively. In the future, we'll support model
   sharding via Zero-3. These methods reduce model state memory by a
   factor of (1 / the number of data-parallel devices).
-  For some models, activation memory can be reduced by setting
   activation checkpoints in the PyTorch module definition.
   *Note that activation checkpointing can reduce
   training throughput by up to 25%.*

Usage
-----

To launch a multi-GPU training job, we provide the ``composer``
launcher:

.. code:: python

   # run training on 8 GPUs
   >>> composer -n 8 my_training_script.py

Under the hood, this script (`source code
here <https://github.com/mosaicml/composer/blob/dev/composer/cli/launcher.py>`__)
sets the required :mod:`torch.distributed` environment variables, launches
the processes, and then runs the script on each process. For additional
configurations of our launcher script, run ``composer --help``.

Distributed Properties
----------------------

Developers may need to access the current rank or world size in a
distributed setting. For example, a callback may only want to log
something for rank zero. Use our :mod:`composer.utils.dist` module to
retrieve this information. The methods are similiar to
:mod:`torch.distributed`, but also return defaults in a non-distributed
setting.

.. code:: python

   from composer.utils import dist

   dist.get_world_size()  # torch.distributed.get_world_size()
   dist.get_local_rank()
   dist.get_global_rank()  # torch.distributed.get_rank()

For all retrievable properties, see :mod:`composer.utils.dist`.

Space-time Equivalence
----------------------

We consider an equivalency principle between distributed training
and gradient accumulation. That is, batches can either be parallelized
across space (e.g. devices) or across time (e.g. gradient accumulation).
Furthermore, the two dimensions are interchangable -- more devices, less gradient
accumulation, and vice versa. Our trainer strives to respect this equivalency
and ensure identical behavior regardless of the combinations of space and time
parallelization used.

..
    TODO: add details on DDP SYNC STRATEGY