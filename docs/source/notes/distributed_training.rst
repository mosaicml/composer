|:family_man_woman_girl_boy:| Distributed Training
==================================================

.. _distributed-training:

Composer supports distributed training on multiple devices, whether it
be multiple GPUs on a single node or multiple GPUs across multiple
nodes.

Data Parallelism
----------------

Composer distributes work across devices via **data-parallelism-only**.
We made this design choice in order to provide the most flexibility to algorithms,
which can modify the training loop in complex ways. Data parallelism
greatly simplifies model building and memory management. Every GPU is
performing the same work, so inspecting the rank zero is sufficient to
reason about memory, performance, and other properties.

Within Composer, we have three options for data-parallelism-only
execution: `Pytorch DDP`_ (default), `Pytorch FSDP`_, and `DeepSpeed Zero`_. Although Pytorch DDP is the default, DeepSpeed Zero provides better performance and lower memory utilization when configured correctly, and Pytorch FSDP increases memory and computational efficiency, while producing the same results as Pytorch DDP.

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

Distributed Sampling
--------------------

When providing :class:`torch.utils.data.Dataset` which is not :class:`torch.utils.data.IterableDataset`
with :class:`torch.utils.data.DataLoader` to Composer, a :class:`torch.utils.data.distributed.DistributedSampler`
is necessary to ensure different devices receive different batches. Composer will
raise an error if a DistributedSampler is not provided. :mod:`composer.utils.dist`
provides a helper function to create a DistributedSampler with the correct
parameters in :meth:`composer.utils.dist.get_sampler`.

.. code:: python

    from composer.utils import dist

    sampler = dist.get_sampler(dataset, shuffle=True)

    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

:class:`composer.datasets.StreamingDataset` is an IterableDataset so a
DistributedSampler is not supported as IterableDatasets need to handle multi-worker
training internally. See IterableDataset [docs](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)
for more information

Deepspeed
---------

Composer comes with DeepSpeed support, allowing you to leverage their
full set of features that makes it easier to train large models across
(1) any type of GPU and (2) multiple nodes. For more details on DeepSpeed,
see `their website <https://www.deepspeed.ai>`__.

We support optimizer and gradient sharing via
`Deepspeed Zero`_ stages 1 and 2 respectively. In the future, we'll support model
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


FullyShardedDataParallel (FSDP)
-------------------------------

Composer integrates Pytorch's `FullyShardedDataParallel <https://pytorch.org/docs/stable/fsdp.html>`__ engine with some syntactic sugar to make it easy to write custom models that work with Composer + FSDP.

At a high level, when you use the Composer Trainer, you must pass it a :mod:`ComposerModel` like `ComposerGPT <https://github.com/mosaicml/examples/blob/6972fe3000d5a5480d8757ff710965514155e8db/llm/llm/gpt.py#L178>`__ that defines certain functions like :code:`forward`, :code:`eval_forward`, :code:`loss`, etc. that are called during the training loop.

Inside that :mod:`ComposerModel` you may have one or many submodules, such as a :code:`.model` or :code:`.language_model` or :code:`.classifier` that is the actual :mod:`torch.nn.Module` that you will be deploying at inference time. In our case, this is the `GPT <https://github.com/mosaicml/examples/blob/6972fe3000d5a5480d8757ff710965514155e8db/llm/llm/gpt.py#L102>`__ module that we build and attach :mod:`ComposerGPT.model`.

When you provide an :code:`fsdp_config={...}` dictionary to the Composer Trainer, then on :code:`__init__`, the Trainer will attempt to wrap **each of the submodules** of your :mod:`ComposerModel` with an FSDP auto wrap policy. This wrapping is recursive, so not only is `GPT` wrapped, but all submodules of `GPT` may/may not be wrapped too. See the `FSDP documentation <https://pytorch.org/docs/stable/fsdp.html>`__ for more details on how auto wrap policies work.

The full spec and defaults for Composer's `fsdp_config` is here:

.. code:: python

    fsdp_config = {
      'sharding_strategy': str = 'FULL_SHARD' | 'SHARD_GRAD_OP' | 'NO_SHARD', # Default: 'FULL_SHARD'
      'min_params': float # Default: 1e8
      'cpu_offload': bool = True | False, # Default: False, cpu_offload not supported yet
      'mixed_precision': str = 'FULL' | 'DEFAULT' | 'PURE', # Default: 'DEFAULT'
      # Note: you can explictly provide a dictionary too
      # 'mixed_precision': dict = {
      #   'param_dtype': 'fp32' | 'fp16' | 'bf16',
      #   'reduce_dtype': 'fp32' | 'fp16' | 'bf16',
      #   'buffer_dtype': 'fp32' | 'fp16' | 'bf16',
      # },
      'backward_prefetch': str = 'BACKWARD_PRE' | 'BACKWARD_POST' | 'NONE', # Default: 'BACKWARD_POST'
      'activation_checkpointing': bool = True | False, # Default: False
      'activation_cpu_offload': bool = True | False, # Default: False
      'verbose': bool = True | False,
    }

All values come with defaults and can be optionally defined in the :code:`fsdp_config`. Most parameters map directly to parameters in the `FSDP documentation <https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel>`__.

One Composer-specific pattern is that if :code:`mixed_precision` is provided as a :code:`str`, then we automatically infer the settings to use from the Trainer's :code:`precision`, which is already being used for autocast, and we construct an associated MixedPrecision object for FSDP:

.. code:: python

    # If mixed_precision = 'full'
    mixed_precision = MixedPrecision(
      param_dtype=torch.float32,
      reduce_dtype=torch.float32,
      buffer_dtype=torch.float32,
    )
    # If mixed_precision = 'default'
    mixed_precision = MixedPrecision(
      param_dtype=torch.float32,
      reduce_dtype=autocast_precision, # Low precision gradient communication
      buffer_dtype=torch.float32,
    )

    # If mixed_precision = 'pure'
    mixed_precision = MixedPrecision(
      param_dtype=autocast_precision, # Low precision master weights
      reduce_dtype=autocast_precision, # Low precision gradient communication
      buffer_dtype=autocast_precision, # Low precision buffers
    )

An example code snippet for using FSDP with composer is provided below:

.. code:: python

    import torch.nn as nn
    from composer import Trainer

    class Block (nn.Module):
        ...

    class Model(nn.Module):
        def __init__(self, n_layers):
            super().__init__()
            self.blocks = nn.ModuleList([
                Block(...) for _ in range(n_layers)
            ]),
            self.head = nn.Linear(...)

        def forward(self, inputs):
            ...

        # FSDP Wrap Function
        def fsdp_wrap_fn(self, module):
            return isinstance(module, Block)

        # Activation Checkpointing Function
        def activation_checkpointing_fn(self, module):
            return isinstance(module, Block)


    class MyComposerModel(ComposerModel):

        def __init__(self, n_layers):
            super().__init__()
            self.model = Model(n_layers)
            ...

        def forward(self, batch):
            ...

        def eval_forward(self, batch, outputs=None):
            ...

        def loss(self, outputs, batch):
            ...

        ...

    composer_model = MyComposerModel(n_layers=3)

    fsdp_config = {
        'sharding_strategy': 'FULL_SHARD',
        'min_params': 1e8,
        'cpu_offload': False, # Not supported yet
        'mixed_precision': 'DEFAULT',
        'backward_prefetch': 'BACKWARD_POST',
        'activation_checkpointing': False,
        'activation_cpu_offload': False,
        'verbose': True
    }


    trainer = Trainer(
        model=composer_model,
        fsdp_config=fsdp_config,
        ...
    )

    trainer.fit()


.. warning::
    As of now now we don't support :code:`CPU Offloading` for FSDP.

.. warning::
    As of now, default parameters might not provide optimal convergence. Please proceed with caution.

Composer's FSDP Auto Wrap Policy
--------------------------------
To make auto-wrapping easier on users, Composer uses a custom auto wrap policy that wraps modules according to the following rules:

1) If any module is attributed with :code:`module._fsdp_wrap = True | False`, that choice will be respected.
2) If the root module (e.g. `GPT`) defines a function :code:`def fsdp_wrap_fn(module: torch.nn.Module) -> bool`, then that function will be used to evaluate the root module's children.
3) If any module has more parameters than :code:`fsdp_config['min_params']`, it will be wrapped.

These rules are meant to make it easy for users to modify existing models for usage with FSDP. You can either add attributes to modules you want to wrap (#1), define a filter (#2), or make no changes at all and just use the size-based policy via :code:`fsdp_config['min_params'] = ...` (#3).

In `gpt.py <https://github.com/mosaicml/examples/blob/6972fe3000d5a5480d8757ff710965514155e8db/llm/llm/gpt.py>`__, you can see that `we used rule #2 <https://github.com/mosaicml/examples/blob/6972fe3000d5a5480d8757ff710965514155e8db/llm/llm/gpt.py#L172>`__ to specify that all :code:`GPTBlock` modules within :code:`GPT` should be wrapped. Alternatively, we could have easily attributed each of the blocks with :code:`block._fsdp_wrap = True` and it would have accomplished the same thing. Whatever style you prefer, it's up to you!

A very similar auto wrap policy is provided for activation checkpointing, with analogous rule #1 that looks for :code:`module._activation_checkpointing = True | False` and rule #2 that looks for :code:`def activation_checkpointing_fn(module: torch.nn.Module) -> bool`.





.. _Pytorch DDP: https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html
.. _Deepspeed Zero: https://www.deepspeed.ai/
.. _Pytorch FSDP: https://pytorch.org/docs/stable/fsdp.html
