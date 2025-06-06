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

Within Composer, we have two options for data-parallelism-only
execution: `Pytorch DDP`_ (default) and `Pytorch FSDP`_.
Although Pytorch DDP is the default, Pytorch FSDP increases memory and computational
efficiency when configured correctly while producing the same results and is the recommended option.

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

By default, only the rank zero logs will be sent to the console. To save the logs
from all the ranks, use ``--stdout`` and ``--stderr``:

.. code:: python

    >>> composer -n 8 --stdout stdout_{rank}.log --stderr stderr_{rank}.log script.py

The stdout for each rank will then be available at ``stdout_1.log``, ``stdout_2.log``, and so forth.
The filename is customizable, see the command help for more details.

Alternatively, the logs can also be captured using our :class:`.FileLogger`.

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
retrieve this information. The methods are similar to
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
Furthermore, the two dimensions are interchangeable -- more devices, less gradient
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


FullyShardedDataParallel (FSDP)
-------------------------------

Composer integrates Pytorch's `FullyShardedDataParallel <https://pytorch.org/docs/stable/fsdp.html>`__
engine with some syntactic sugar to make it easy to write custom models that work with Composer + FSDP.

At a high level, when you use the Composer Trainer, you must pass it a :mod:`ComposerModel` like
`ComposerGPT <https://github.com/mosaicml/examples/blob/6972fe3000d5a5480d8757ff710965514155e8db/llm/llm/gpt.py#L178>`__
that defines certain functions like :code:`forward`, :code:`eval_forward`, :code:`loss`, etc. that
are called during the training loop.

Inside that :mod:`ComposerModel` you may have one or many submodules, such as a :code:`.model` or
:code:`.language_model` or :code:`.classifier` that is the actual :mod:`torch.nn.Module` that you
will be deploying at inference time. In our case, this is the
`GPT <https://github.com/mosaicml/examples/blob/6972fe3000d5a5480d8757ff710965514155e8db/llm/llm/gpt.py#L102>`__
module that we build and attach :mod:`ComposerGPT.model`.

When you provide an :code:`parallelism_config={'fsdp': {...}}` dictionary to the Composer Trainer,
then on :code:`__init__`, the Trainer will attempt to wrap **each of the submodules** of your
:mod:`ComposerModel` with an FSDP auto wrap policy. This wrapping is recursive, so not only is
`GPT` wrapped, but all submodules of `GPT` may/may not be wrapped too. See the
`FSDP documentation <https://pytorch.org/docs/stable/fsdp.html>`__ for more details on how auto
wrap policies work.

The full spec and defaults for Composer's fsdp config is here:

.. code:: python

    fsdp_config = {
      'activation_checkpointing': bool = True | False, # Default: False
      'activation_checkpointing_reentrant': bool = True | False, # Default: True
      'activation_cpu_offload': bool = True | False, # Default: False
      'backward_prefetch': str = 'BACKWARD_PRE' | 'BACKWARD_POST' | 'NONE', # Default: 'BACKWARD_POST'
      'cpu_offload': bool = True | False, # Default: False, cpu_offload not supported yet
      'data_parallel_shard_degree': int = -1, # Default: -1
      'data_parallel_replicate_degree': int = 1, # Default: 1
      'forward_prefetch': bool = True | False, # Default: False
      'ignored_modules': Optional[Iterable[torch.nn.Module]], # Default: None
      'keep_low_precision_grads': bool = True | False, # Default: False
      'limit_all_gathers': bool = True | False, # Default: False
      'load_monolith_rank0_only': bool = True | False, # Default: False
      'load_planner': torch.distributed.checkpoint.planner.LoadPlanner, # Default: None
      'mixed_precision': str = 'FULL' | 'DEFAULT' | 'PURE', # Default: 'DEFAULT'
      # Note: you can explicitly provide a dictionary too
      # 'mixed_precision': dict = {
      #   'param_dtype': 'fp32' | 'fp16' | 'bf16',
      #   'reduce_dtype': 'fp32' | 'fp16' | 'bf16',
      #   'buffer_dtype': 'fp32' | 'fp16' | 'bf16',
      # },
      'save_planner': torch.distributed.checkpoint.planner.SavePlanner, # Default: None
      'sharded_ckpt_prefix_dir': str = 'ep{epoch}-ba{batch}', # Default: 'ep{epoch}-ba{batch}'
      'sharding_strategy': str = 'FULL_SHARD' | 'SHARD_GRAD_OP' | 'NO_SHARD', # Default: 'FULL_SHARD'
      'state_dict_type': str = 'full' | 'local' | 'sharded', # Default: full
      'sync_module_states': bool = True | False, # Default: False
      'use_orig_params': bool = True | False, # Default: True
      'verbose': bool = True | False, # Default: False
    }

All values come with defaults and can be optionally defined in the :code:`fsdp_config`. Most
parameters map directly to parameters in the
`FSDP documentation <https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel>`__.
This config is passed under `parallelism_config['fsdp']` to the Composer Trainer. Two important
parameters which do not map include `data_parallel_shard_degree`, which dictates the number of
devices to shard across, and `data_parallel_replicate_degree`, which dictates the number of
devices to replicate across.

One Composer-specific pattern is that if :code:`mixed_precision` is provided as a :code:`str`,
then we automatically infer the settings to use from the Trainer's :code:`precision`, which is
already being used for autocast, and we construct an associated MixedPrecision object for FSDP:

.. code:: python

    # If mixed_precision = 'full'
    mixed_precision = MixedPrecision(
      param_dtype=torch.float32,
      reduce_dtype=torch.float32,
      buffer_dtype=torch.float32,
    )
    # If mixed_precision = 'default'; emulates automatic mixed precision training.
    mixed_precision = MixedPrecision(
      param_dtype=autocast_precision,  # Master weights stored in fp32 but are downcast to autocast_precision before the dist all_gather
      reduce_dtype=torch.float32,  # Gradient dist all_reduce in fp32
      buffer_dtype=autocast_precision,  # Buffers stored in fp32 but are downcast to autocast_precision before the dist all_gather
    )
    # If mixed_precision = 'pure'
    mixed_precision = MixedPrecision(
      param_dtype=autocast_precision,  # Master weights stored in fp32 but are downcast to autocast_precision before the dist all_gather
      reduce_dtype=autocast_precision,  # Gradient dist all_reduce in autocast_precision
      buffer_dtype=autocast_precision,  # Buffers stored in fp32 but are downcast to autocast_precision before the dist all_gather
    )

An example code snippet for using FSDP with composer is provided below:

.. code:: python

    import torch.nn as nn
    from composer import Trainer

    class Block(nn.Module):
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
        'cpu_offload': False, # Not supported yet
        'mixed_precision': 'DEFAULT',
        'backward_prefetch': 'BACKWARD_POST',
        'activation_checkpointing': False,
        'activation_cpu_offload': False,
        'verbose': True
    }


    trainer = Trainer(
        model=composer_model,
        parallelism_config={'fsdp': fsdp_config},
        ...
    )

    trainer.fit()


.. warning::
    As of now we don't support :code:`CPU Offloading` for FSDP.

.. warning::
    As of now, default parameters might not provide optimal convergence. Please proceed with caution.

Composer's FSDP Auto Wrap Policy
--------------------------------
To make auto-wrapping easier on users, Composer uses a custom auto wrap policy that wraps modules according to the following rules:

1) If any module is attributed with :code:`module._fsdp_wrap = True | False`, that choice will be respected.
2) If the root module (e.g. `GPT`) defines a function :code:`def fsdp_wrap_fn(module: torch.nn.Module) -> bool`, then that function will be used to evaluate the root module's children.

These rules are meant to make it easy for users to modify existing models for usage with FSDP. You can either add attributes to modules you want to wrap (#1) or define a filter (#2).

In `gpt.py <https://github.com/mosaicml/examples/blob/6972fe3000d5a5480d8757ff710965514155e8db/llm/llm/gpt.py>`__, you can see that `we used rule #2 <https://github.com/mosaicml/examples/blob/6972fe3000d5a5480d8757ff710965514155e8db/llm/llm/gpt.py#L172>`__ to specify that all :code:`GPTBlock` modules within :code:`GPT` should be wrapped. Alternatively, we could have easily attributed each of the blocks with :code:`block._fsdp_wrap = True` and it would have accomplished the same thing. Whatever style you prefer, it's up to you!

A very similar auto wrap policy is provided for activation checkpointing, with analogous rule #1 that looks for :code:`module._activation_checkpointing = True | False` and rule #2 that looks for :code:`def activation_checkpointing_fn(module: torch.nn.Module) -> bool`.


**Experimental:** Composer enables users to specify custom FSDP args for all wrapped modules. This is enabled by returning a dictionary of args instead of returning a bool.

.. code:: python

    import torch.nn as nn

    class Block(nn.Module):
        ...

    class BlockRequiringCustomArgs(nn.Module):
        ...

    class Model(nn.Module):
        def __init__(self, n_layers):
            super().__init__()
            self.blocks = nn.ModuleList([
                Block(...) for _ in range(n_layers)
            ])
            self.custom_arg_blocks = nn.ModuleList([
                BlockRequiringCustomArgs(...) for _ in range(n_layers)
            ]),
            self.head = nn.Linear(...)

        def forward(self, inputs):
            ...

        # FSDP Wrap function
        def fsdp_wrap_fn(self, module):
            if isinstance(module, Block):
                return True

            # extends FSDP wrapping to custom args
            if isinstance(module, BlockRequiringCustomArgs):
                return {
                    'process_group': 'node',
                    'mixed_precision': 'FULL',
                }

            # default to False
            return False

        # Activation Checkpointing Function
        def activation_checkpointing_fn(self, module):
            return isinstance(module, Block)

While the user can instantiate and pass in process groups, Composer enables process groups to be
specified using the following options:

1. :code:`'self'`: the degenerate case where all process groups only operate within their current rank (:code:`'self'` == :code:`'set1'`). This is useful when you do not want a layer to be synchronized across accelerators.

2. :code:`'node'`: instantiates process groups which operate within a node (:code:`'node'` == :code:`f'set{local_world_size}'`). This is useful for Expert Layers in MoE models.

3. :code:`'local_rank_across_nodes'`: instantiates process groups with the same local rank across all nodes  (:code:`'local_rank_across_nodes'` == :code:`f'mod{local_world_size}'`). This is useful for Tensor Parallel Layers.

4. :code:`'setK'`: (:code:`K` is an integer where world_size must be divisible by :code:`K`) instantiates process groups which operate within a set of K GPUs. This is useful for Expert Layers in MoE models.

5. :code:`'modK'`: (:code:`K` is an integer where world_size must be divisible by :code:`K`) instantiates process groups which operate on every Kth GPUs. This is useful for Tensor Parallel Layers.


Saving and Loading Sharded Checkpoints with FSDP
------------------------------------------------
To save and load sharded checkpoints with FSDP, you can make use of the field, :code:`state_dict_type` in :code:`fsdp_config`.
Depending on the value you set for :code:`state_dict_type`, you can get different checkpointing behavior:

1. :code:`state_dict_type='full'`
The default. Saves one big checkpoint file for the whole model.
It does this by gathering the model state to the global rank 0 device, unflattening it, and then saving it out.
If `load_monolith_rank0_only=True`, then when loading checkpoints the global rank 0 device will load
in the checkpoint file and scatter the model and optimizer state to the other ranks, which will
dramatically reduce the memory usage on system. Otherwise, all ranks will separately load in the checkpoint file.

2. :code:`state_dict_type='sharded'`
Each rank saves out an unflattened shard. For loading, each rank loads in the checkpoint file
corresponding to their unflattened shard.
**Note: state_dict_type='sharded' is the recommended setting for sharded checkpointing in Composer for torch versions 2.0.0 or higher.**

See `The FSDP docs <https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.state_dict>`__ for more info.

If you use sharded checkpoints (`state_dict_type='sharded'`), your run will save as many files as you have
ranks at each checkpointing event (plus one metadata file for torch versions 2.0.0 or higher). This can quicky
pollute your `save_folder` with a lot of files after a couple checkpointing events. To help keep your
checkpoint shard files organized, Composer will save each set of shards in its own prefix directory, which you can configure
by using `'sharded_ckpt_prefix_dir'` (default value `sharded_ckpt_prefix_dir='ep{epoch}-ba{batch}'`). Checkpoint shards will be saved to
`{save_folder} / {sharded_ckpt_prefix_dir}`

For example, to save sharded checkpoints to disk locally (`state_dict_type='sharded'`) with FSDP on PyTorch version 2.0.0 and higher, you can do:

.. code:: python

    import torch.nn as nn
    from composer import Trainer

    class Block(nn.Module):
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
        'state_dict_type': 'sharded',
        'sharded_ckpt_prefix_dir': 'ba{batch}-shards' # will save each set of shards checkpoint to a unique folder based on batch

    }

    trainer = Trainer(
        model=composer_model,
        max_duration='4ba'
        parallelism_config={'fsdp': fsdp_config},
        save_folder='checkpoints',
        save_interval='2ba',
        ...
    )

    trainer.fit()

After the second batch, this code will save N+1 checkpoint files to the local directory ``./checkpoints/ba2-shards``. For example,
if you trained with 4 ranks, ``./checkpoints/ba2-shards`` would contain 5 files: a metadata file: ``.metadata`` and 4 checkpoint files for each rank: ``__0_0.distcp``, ``__1_0.distcp``, ``__2_0.distcp``, and ``__3_0.distcp``.
After the fourth batch, N+1 checkpoint files (``.metadata``, ``__0_0.distcp``, ``__1_0.distcp``, etc.) will saved to ``./checkpoints/ba4-shards``
To load these checkpoint files, you would need to do something like this:

.. code:: python

    from composer import Trainer

    fsdp_config = {
        'sharding_strategy': 'FULL_SHARD',
        'state_dict_type': 'sharded',
    }


    trainer = Trainer(
        model=composer_model,
        max_duration='4ba'
        parallelism_config={'fsdp': fsdp_config},
        load_path='./checkpoints/ba2-shards' # load_path must be the path to the prefix directory and not to a specific file.
        ...
    )

Four things to note in this load example:

1. Instead of setting ``load_path`` to the path to a specific file, we set it to the directory which contains all the checkpoint files.

2. We must set ``'state_dict_type': 'sharded'``, like we did during the save.

3. Composer with PyTorch version 2.0.0 and higher **does** support elastic checkpointing (more ranks than checkpoint files or more files than ranks), so you can resume on a different number of ranks than you saved on.

4. To do multinode resumption (resuming on more than one node regardless of how many nodes you saved on), you must be using torch 2.0.1 or higher due to a bug in torch 2.0.0.


Tensor Parallel (TP)
--------------------

Composer integrates Pytorch's `Tensor Parallel <https://pytorch.org/docs/stable/distributed.tensor.parallel.html>`__
API with some syntactic sugar to make it easy to write custom models that work with Composer + TP.

To enable Tensor Parallel, a tensor parallel config must be passed to the Composer Trainer. The
full spec and defaults for Composer's tensor parallelism config is here:

.. code:: python

    tp_config = {
        tensor_parallel_degree: int = 1, # Default: 1
        layer_plan: dict = None, # Default: None, maps to torch's `parallelize_plan`
    }

All values come with defaults and can be optionally defined in the :code:`tp_config`. Most parameters
map directly to parameters in the
`Tensor Parallel documentation <https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.parallelize_module>`__.
This config is passed under `parallelism_config['tp']` to the Composer Trainer. Important parameters
which do not directly map include `tensor_parallel_degree`, which dictates the number of devices to shard across,
and `layer_plan`, which simply corresponds to torch's `parallelize_plan`.


An example code snippet for using TP and FSDP with Composer is provided below:

.. code:: python

    import torch.nn as nn
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
    from composer import Trainer

    class Block(nn.Module):
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
        'cpu_offload': False, # Not supported yet
        'mixed_precision': 'DEFAULT',
        'backward_prefetch': 'BACKWARD_POST',
        'activation_checkpointing': False,
        'activation_cpu_offload': False,
        'verbose': True
    }
    tp_config = {
        'tensor_parallel_degree': 2,
        layer_plan = {
            'model.0.fc1': ColwiseParallel(),
            'model.0.fc2': RowwiseParallel(),
        }
    }

    trainer = Trainer(
        model=composer_model,
        parallelism_config={
            'fsdp': fsdp_config,
            'tp': tp_config,
        },
        ...
    )

    trainer.fit()

.. note::
    This is an experimental feature and is subject to change. Many features, such as `load_monolith_rank0_only` or tensor parallelism without FSDP, are not yet supported.

.. _Pytorch DDP: https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html
.. _Pytorch FSDP: https://pytorch.org/docs/stable/fsdp.html
