|:family_man_woman_girl_boy:| Distributed Training
==================================================

.. _distributed-training:

Composer supports distributed training on multiple devices, whether it
be multiple GPUs on a single node or multiple GPUs across multiple
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
execution: `Pytorch DDP`_` and `DeepSpeed Zero`_. We currently default to
Pytorch DDP, though DeepSpeed Zero can provide better performance and
lower memory utilization when configured correctly.


Usage
-----

To launch a multi-GPU training job, we provide the ``composer`` launcher:

.. code:: python

    # run training on 8 GPUs
    >>> composer -n 8 my_training_script.py

Under the hood, this script (`source code
here <https://github.com/mosaicml/composer/blob/dev/composer/cli/launcher.py>`__)
sets the required :mod:`torch.distributed` environment variables, launches
the processes, and runs the script on each process.

.. note::
    The ``batch_size`` passed to your dataloader should be the per-device
    *mini*\ batch size. We further split this into smaller microbatches with
    gradient accumulation.


For additional configurations of our launcher script, run ``composer --help``.

.. argparse::
   :module: composer.cli.launcher
   :func: _get_parser
   :prog: composer
   :nodescription:


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

..
    TODO: add details on DDP SYNC STRATEGY

Space-time Equivalence
----------------------

We consider an equivalency principle between distributed training
and gradient accumulation. That is, batches can either be parallelized
across space (e.g. devices) or across time (e.g. gradient accumulation).
Furthermore, the two dimensions are interchangable -- more devices, less gradient
accumulation, and vice versa. Our trainer strives to respect this equivalency
and ensure identical behavior regardless of the combinations of space and time
parallelization used.


Deepspeed
---------

Composer comes with DeepSpeed support, allowing you to leverage their
full set of features that makes it easier to train large models across
(1) any type of GPU and (2) multiple nodes. For more details on DeepSpeed,
see `their website <https://www.deepspeed.ai>`__.

We support optimizer and gradient sharing via
`Deepspeed Zero`_` stages 1 and 2 respectively. In the future, we'll support model
sharding via Zero-3. These methods reduce model state memory by a
factor of (1 / the number of data-parallel devices).

To enable DeepSpeed, simply pass in a config as specified in the
DeepSpeed docs `here <https://www.deepspeed.ai/docs/config-json/>`__.

.. code:: python

    # run_trainer.py

    from composer import Trainer

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_duration='160ep',
        device='gpu',
        deepspeed_config={
            "train_batch_size": 2048,
            "fp16": {"enabled": True},
        })

Providing an empty dictionary to deepspeed is also valid. The deepspeed
defaults will be used and other fields (such as precision) will be inferred
from the trainer.

.. warning::

    The ``deepspeed_config`` must not conflict with any other parameters
    passed to the trainer.

.. warning::

    Not all algorithms have been tested with Deepspeed, please proceed with
    caution.

.. _Pytorch DDP: https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html
.. _Deepspeed Zero: https://www.deepspeed.ai/
