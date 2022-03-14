# Copyright 2021 MosaicML. All Rights Reserved.

"""Train models!

The trainer supports models with :class:`~composer.models.base.ComposerModel` instances.
The :class:`.Trainer` is highly customizable and can
support a wide variety of workloads.

Example
--------

Train a model and save a checkpoint:

.. testcode::

    import os
    from composer import Trainer

    ### Create a trainer
    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      max_duration="1ep",
                      eval_dataloader=eval_dataloader,
                      optimizers=optimizer,
                      schedulers=scheduler,
                      device="cpu",
                      validate_every_n_epochs=1,
                      save_folder="checkpoints",
                      save_interval="1ep")

    ### Fit and run evaluation for 1 epoch.
    ### Save a checkpoint after 1 epoch as specified during trainer creation.
    trainer.fit()

Load the checkpoint and resume training:

.. testcode::

    ### Get the saved checkpoint folder
    ### By default, the checkpoint folder is of the form runs/<timestamp>/rank_0/checkpoints
    ### Alternatively, if you set the run directory environment variable as follows:
    ### os.environ["COMPOSER_RUN_DIRECTORY"] = "my_run_directory", then the checkpoint path
    ### will be of the form my_run_directory/rank_0/checkpoints
    checkpoint_folder = trainer.checkpoint_folder

    ### If the save_interval was in terms of epochs like above then by default,
    ### checkpoint filenames are of the form "ep{EPOCH_NUMBER}.pt".
    checkpoint_path = os.path.join(checkpoint_folder, "ep1.pt")

    ### Create a new trainer with the load_path_format argument set to the checkpoint path.
    ### This will automatically load the checkpoint on trainer creation.
    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      max_duration="2ep",
                      eval_dataloader=eval_dataloader,
                      optimizers=optimizer,
                      schedulers=scheduler,
                      device="cpu",
                      validate_every_n_epochs=1,
                      load_path_format=checkpoint_path)

    ### Continue training and running evaluation where the previous trainer left off
    ### until the new max_duration is reached.
    ### In this case it will be one additional epoch to reach 2 epochs total.
    trainer.fit()
"""

from __future__ import annotations

import contextlib
import datetime
import itertools
import logging
import textwrap
import warnings
from typing import Any, Callable, ContextManager, Dict, List, Optional, Sequence, Union, cast

import torch
import torch.distributed
import torch.utils.data
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric

import composer
from composer.algorithms import ScaleSchedule
from composer.callbacks import CheckpointSaver
from composer.core import Callback, DataSpec, Engine, Event, Logger, State, Time
from composer.core.algorithm import Algorithm
from composer.core.evaluator import Evaluator
from composer.core.logging import LoggerCallback, LogLevel
from composer.core.time import Timestamp
from composer.core.types import (Batch, BreakEpochException, DataLoader, Evaluators, Many, Metrics, Optimizers,
                                 Precision, PyTorchScheduler)
from composer.datasets.dataloader import unwrap_data_loader
from composer.loggers.tqdm_logger import TQDMLogger
from composer.models.base import ComposerModel
from composer.optim.decoupled_weight_decay import DecoupledSGDW
from composer.optim.scheduler import ComposerScheduler, compile_composer_scheduler
from composer.profiler import Profiler, ProfilerEventHandler
from composer.profiler.dataloader_profiler import DataLoaderProfiler
from composer.profiler.system_profiler import SystemProfiler
from composer.profiler.torch_profiler import TorchProfiler
from composer.trainer._deepspeed import _fix_batch_precision_for_deepspeed, _parse_deepspeed_config
from composer.trainer._scale_schedule import scale_pytorch_scheduler
from composer.trainer._scaler import ClosureGradScaler
from composer.trainer.ddp import DDPSyncStrategy, _ddp_sync_context, _prepare_ddp_module
from composer.trainer.devices import Device, DeviceCPU, DeviceGPU
from composer.utils import dist, ensure_tuple, map_collection, module_surgery, reproducibility
from composer.utils.checkpoint import load_checkpoint, save_checkpoint
from composer.utils.object_store import ObjectStoreProvider

log = logging.getLogger(__name__)

__all__ = ["Trainer"]


class Trainer:
    """Trainer for training a models with Composer algorithms. See the Trainer guide for more information.

    Args:
        model (ComposerModel): The model to train. Can be user-defined or one of the models included
            with Composer.

            .. seealso:: :mod:`composer.models` for models built into Composer.
        train_dataloader (DataLoader, DataSpec, or dict): The :class:`.DataLoader`, :class:`.DataSpec`,
            or dict of :class:`.DataSpec` kwargs for the training data. In order to specify custom
            preprocessing steps on each data batch, specify a :class:`.DataSpec` instead of a
            :class:`.DataLoader`.

            .. note:: The ``train_dataloader`` should yield per-rank batches. Each per-rank batch
                will then be further divided based on the ``grad_accum`` parameter. For example, if the
                desired optimization batch size is ``2048`` and training is happening across 8 GPUs, then each
                ``train_dataloader`` should yield a batch of size ``2048 / 8 = 256``. If ``grad_accum = 2``,
                then the per-rank batch will be divided into microbatches of size ``256 / 2 = 128``.
        max_duration (int, str, or Time): The maximum duration to train. Can be an integer, which will be
            interpreted to be epochs, a str (e.g. ``1ep``, or ``10ba``), or a :class:`.Time` object.
        eval_dataloader (DataLoader, DataSpec, or Evaluators, optional): The :class:`.DataLoader`,
            :class:`.DataSpec`, or :class:`.Evaluators` for the evaluation data.
            In order to evaluate one or more specific metrics across one or more datasets, pass in an
            :class:`.Evaluator`. If a :class:`.DataSpec` or :class:`.DataLoader` is passed in, then all
            metrics returned by ``model.metrics()`` will be used during evaluation.
            ``None`` results in no evaluation. (default: ``None``)
        algorithms (List[Algorithm], optional): The algorithms to use during training. If ``None``, then
            no algorithms will be used. (default: ``None``)

            .. seealso:: :mod:`composer.algorithms` for the different algorithms built into Composer.
        optimizers (Optimizers, optional): The optimizer.
            If ``None``, will be set to ``DecoupledSGDW(model.parameters(), lr=0.1)``. (default: ``None``)

            .. seealso:: :mod:`composer.optim` for the different optimizers built into Composer.
        schedulers (Schedulers, optional): The learning rate schedulers. If ``[]`` or ``None``, will be set to
            ``[constant_scheduler]``. (default: ``None``).

            .. seealso:: :mod:`composer.optim.scheduler` for the different schedulers built into Composer.
        device (str or Device, optional): The device to use for training. Either ``cpu`` or ``gpu``.
            (default: ``cpu``)
        grad_accum (int, optional): The number of microbatches to split a per-device batch into. Gradients
            are summed over the microbatches per device. (default: ``1``)

            .. note:: This is implemented by taking the batch yielded by the ``train_dataloader`` and splitting
                it into ``grad_accum`` sections. Each section is of size ``train_dataloader // grad_accum``.
                If the batch size of the dataloader is not divisible by ``grad_accum``,
                then the last section will be of size ``batch_size % grad_accum``.
        grad_clip_norm (float, optional): The norm to clip gradient magnitudes to. Set to ``None`` for no gradient
            clipping. (default: ``None``)
        validate_every_n_batches (int, optional): Compute metrics on evaluation data every N batches.
             Set to ``-1`` to never validate on a batchwise frequency. (default: ``-1``)
        validate_every_n_epochs (int, optional): Compute metrics on evaluation data every N epochs.
            Set to ``-1`` to never validate on a epochwise frequency. (default: ``1``)
        compute_training_metrics (bool, optional): ``True`` to compute metrics on training data and ``False`` to not.
            (default: ``False``)
        precision (str or Precision, optional): Numerical precision to use for training. One of ``fp32``, ``fp16``
            or ``amp`` (recommended). (default: ``Precision.FP32``)

            .. note::
                ``fp16`` only works if ``deepspeed_config`` is also provided.
        scale_schedule_ratio (float, optional): Ratio by which to scale the training duration and learning rate
            schedules. E.g., ``0.5`` makes the schedule take half as many epochs and ``2.0`` makes it take twice as
            many epochs. ``1.0`` means no change. (default: ``1.0``)

            .. note ::

                Training for less time, while rescaling the learning rate schedule,
                is a strong baseline approach to speeding up training. E.g., training
                for half duration often yields minor accuracy degradation,
                provided that the learning rate schedule is also rescaled to take half as long.

                To see the difference, consider training for half as long using a cosine
                annealing learning rate schedule. If the schedule is not rescaled,
                training ends while the learning rate is still ~0.5 of the initial LR.
                If the schedule is rescaled with ``scale_schedule_ratio``, the LR schedule
                would finish the entire cosine curve, ending with a learning rate near zero.

        step_schedulers_every_batch (bool, optional): By default, native
            `PyTorch schedulers <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_
            are updated every epoch, while :doc:`Composer Schedulers</trainer/schedulers>` are updated every step.
            Setting this to ``True`` will force schedulers to be stepped every batch,
            while ``False`` means schedulers stepped every epoch. ``None`` indicates the default behavior.
            (default: ``None``)
        dist_timeout (float, optional): Timeout, in seconds, for initializing the distributed process group.
            (default: ``15.0``)
        ddp_sync_strategy (str or DDPSyncStrategy, optional): The strategy to use for synchronizing gradients.
            Leave unset to let the trainer auto-configure this. See :class:`.DDPSyncStrategy`
            for more details.
        seed (int, optional): The seed used in randomization. If ``None``, then a random seed
            will be created. (default: ``None``)

            .. note:: In order to get reproducible results, call the
                :func:`.seed_all` function at the start of your script with the seed
                passed to the trainer. This will ensure any initialization done before the trainer init
                (ex. model weight initialization) also uses the provided seed.

            .. seealso:: :mod:`composer.utils.reproducibility` for more details on reproducibility.
        deterministic_mode (bool, optional): Run the model deterministically. (default: ``False``)

            .. note:: This is an experimental feature. Performance degradations expected. Certain Torch modules may
                not have deterministic implementations, which will result in a crash.

            .. note:: In order to get reproducible results, call the
                :func:`.configure_deterministic_mode` function at the start of your script.
                This will ensure any initialization done before the trainer init also runs deterministically.

            .. seealso:: :mod:`composer.utils.reproducibility` for more details on reproducibility.
        loggers (Sequence[LoggerCallback], optional): The destinations to log training information to.
            If ``None``, will be set to ``[TQDMLogger()]``. (default: ``None``)

            .. seealso:: :mod:`composer.loggers` for the different loggers built into Composer.
        callbacks (Sequence[Callback], optional): The callbacks to run during training. If ``None``,
            then no callbacks will be run. (default: ``None``).

            .. seealso:: :mod:`composer.callbacks` for the different callbacks built into Composer.
        load_path_format (str, optional):  The path format string to an existing checkpoint file.

            It can be a path to a file on the local disk, a URL, or if ``load_object_store`` is set, the object name
            for a checkpoint in a cloud bucket.

            When using `Deepspeed ZeRO <https://www.deepspeed.ai/tutorials/zero/>`_, checkpoints are shareded by rank.
            Instead of hard-coding the rank in the ``path_format``, use the following format variables:

            +------------------------+-------------------------------------------------------+
            | Variable               | Description                                           |
            +========================+=======================================================+
            | ``{rank}``             | The global rank, as returned by                       |
            |                        | :func:`~.dist.get_global_rank`.                       |
            +------------------------+-------------------------------------------------------+
            | ``{local_rank}``       | The local rank of the process, as returned by         |
            |                        | :func:`~.dist.get_local_rank`.                        |
            +------------------------+-------------------------------------------------------+
            | ``{node_rank}``        | The node rank, as returned by                         |
            |                        | :func:`~.dist.get_node_rank`.                         |
            +------------------------+-------------------------------------------------------+

            For example, suppose that checkpoints are stored in the following structure:

            .. code-block::

                my_model/ep1-rank0.tar
                my_model/ep1-rank1.tar
                my_model/ep1-rank2.tar
                ...

            Then, ``load_path_format`` should be set to ``my_model/ep1-rank{rank}.tar``, and all ranks will load the
            correct state.

            If ``None`` then no checkpoint will be loaded. (default: ``None``)
        load_object_store (ObjectStoreProvider, optional): If the ``load_path_format`` is in an object store
            (i.e. AWS S3 or Google Cloud Storage), an instance of :class:`.ObjectStoreProvider` which
            will be used to retreive the checkpoint. Otherwise, if the checkpoint is a local filepath,
            set to ``None``. Ignored if ``load_path_format`` is ``None``. (default: ``None``)

            Example:

            .. testsetup::

                import composer.trainer

                composer.trainer.trainer.load_checkpoint = lambda *args, **kwargs: None

            .. testcode::

                from composer import Trainer
                from composer.utils import ObjectStoreProvider

                # Create the object store provider with the specified credentials
                creds = {"key": "object_store_key",
                         "secret": "object_store_secret"}
                store = ObjectStoreProvider(provider="s3",
                                            container="my_container",
                                            provider_init_kwargs=creds)

                checkpoint_path = "/path_to_the_checkpoint_in_object_store"

                # Create a trainer which will load a checkpoint from the specified object store
                trainer = Trainer(model=model,
                                  train_dataloader=train_dataloader,
                                  max_duration="10ep",
                                  eval_dataloader=eval_dataloader,
                                  optimizers=optimizer,
                                  schedulers=scheduler,
                                  device="cpu",
                                  validate_every_n_epochs=1,
                                  load_path_format=checkpoint_path,
                                  load_object_store=store)


        load_weights_only (bool, optional): Whether or not to only restore the weights from the checkpoint without
            restoring the associated state. Ignored if ``load_path_format`` is ``None``. (default: ``False``)
        load_strict (bool, optional): Ensure that the set of weights in the checkpoint and model must exactly match.
            Ignored if ``load_path_format`` is ``None``. (default: ``False``)
        load_chunk_size (int, optional): Chunk size (in bytes) to use when downloading checkpoints.
            Ignored if ``load_path_format`` is either ``None`` or a local file path. (default: ``1,048,675``)
        load_progress_bar (bool, optional): Display the progress bar for downloading the checkpoint.
            Ignored if ``load_path_format`` is either ``None`` or a local file path. (default: ``True``)
        save_folder (str, optional): Folder where checkpoints are saved. If ``None``, checkpoints will not be saved
            by default.
            .. seealso:: :class:`~.CheckpointSaver`

            .. note::

                For fine-grained control on checkpoint saving (e.g. to save different types of checkpoints
                at different intervals), leave this parameter as ``None``, and instead pass
                instance(s) of :class:`~.CheckpointSaver` directly as ``callbacks``.

            (default: ``None``)

        save_name_format (str, optional): A format string describing how to name checkpoints.
            This parameter has no effect if ``save_folder`` is ``None``.
            (default: ``"ep{epoch}-ba{batch}-rank{rank}"``)

            .. seealso:: :class:`~.CheckpointSaver`

        save_latest_format (str, optional): A format string for the name of a symlink
            (relative to ``checkpoint_folder``) that points to the last saved checkpoint.
            This parameter has no effect if ``save_folder`` is ``None``.
            To disable symlinking, set to ``None``. (default: ``"latest-rank{rank}"``)

            .. seealso:: :class:`~.CheckpointSaver`

        save_overwrite (bool, optional): Whether existing checkpoints should be overridden.
            This parameter has no effect if ``save_folder`` is None. (default: ``False``)

            .. seealso:: :class:`~.CheckpointSaver`

        save_interval (Time | str | int | (State, Event) -> bool): A :class:`Time`, time-string, integer (in epochs),
            or a function that takes (state, event) and returns a boolean whether a checkpoint should be saved.
            This parameter has no effect if ``save_folder`` is ``None``. (default: ``'1ep'``)

            .. seealso:: :class:`~.CheckpointSaver`

        save_weights_only (bool, optional): Whether to save only the model weights instead of the entire training
            state. This parameter has no effect if ``save_folder`` is ``None``. (default: ``False``)

            .. seealso:: :class:`~.CheckpointSaver`

        train_subset_num_batches (int, optional): If specified, finish every epoch early after training
            on this many batches. This parameter has no effect if it is greater than ``len(train_dataloader)``.
            If ``None``, then the entire dataloader will be iterated over. (default: ``None``)
        eval_subset_num_batches (int, optional): If specified, evaluate on this many batches.
            This parameter has no effect if it is greater than ``len(eval_dataloader)``.
            If ``None``, then the entire dataloader will be iterated over. (default: ``None``)
        deepspeed_config (bool or Dict[str, Any], optional): Configuration for DeepSpeed, formatted as a JSON
            according to `DeepSpeed's documentation <https://www.deepspeed.ai/docs/config-json/>`_. If ``True`` is
            provided, the trainer will initialize the DeepSpeed engine with an empty config ``{}``. If ``False``
            is provided, deepspeed will not be used. (default: ``False``)
        profiler_trace_file (str, optional): Name of the trace file, relative to the run directory.
            Setting this parameter activates the profiler. (default: ``None``).

            .. seealso:: :mod:`composer.profiler` for more details on profiling with the trainer.
        prof_event_handlers (List[ProfilerEventHandler], optional): Trace event handler.
            Ignored if ``profiler_trace_file`` is not specified. (default: ``[JSONTraceHandler()]``).
        prof_skip_first (int, optional): Number of batches to skip at epoch start.
            Ignored if ``profiler_trace_file`` is not specified. (default: ``0``).
        prof_wait (int, optional): Number of batches to skip at the beginning of each cycle.
            Ignored if ``profiler_trace_file`` is not specified. (default: ``0``).
        prof_warmup (int, optional): Number of warmup batches in a cycle.
            Ignored if ``profiler_trace_file`` is not specified. (default: ``1``).
        prof_active (int, optional): Number of batches to profile in a cycle.
            Ignored if ``profiler_trace_file`` is not specified. (default: ``4``).
        prof_repeat (int, optional): Maximum number of profiling cycle repetitions per epoch (0 for no maximum).
            Ignored if ``profiler_trace_file`` is not specified. (default: ``1``).
        sys_prof_cpu (bool, optional): Whether to record cpu statistics.
            Ignored if ``profiler_trace_file`` is not specified. (default: ``True``).
        sys_prof_memory (bool, optional): Whether to record memory statistics.
            Ignored if ``profiler_trace_file`` is not specified. (default: ``False``).
        sys_prof_disk (bool, optional): Whether to record disk statistics.
            Ignored if ``profiler_trace_file`` is not specified. (default: ``False``).
        sys_prof_net (bool, optional): Whether to record network statistics.
            Ignored if ``profiler_trace_file`` is not specified. (default: ``False``).
        sys_prof_stats_thread_interval_seconds (float, optional): Interval to record stats, in seconds.
            Ignored if ``profiler_trace_file`` is not specified. (default: ``0.5``).
        torch_profiler_trace_dir (str, optional): Directory to store trace results relative to the run directory.
            Must be specified to activate the Torch profiler. Ignored if ``profiler_trace_file`` is not specified.
            See :mod:`~composer.profiler`.  (default: ``None``).
        torch_prof_use_gzip (bool): Whether to use gzip for trace.
            Ignored if ``torch_profiler_trace_dir`` and ``profiler_trace_file`` are not specified. (default: ``False``).
        torch_prof_record_shapes (bool, optional): Whether to record tensor shapes.
            Ignored if ``torch_profiler_trace_dir`` and ``profiler_trace_file`` are not specified. (default: ``False``).
        torch_prof_profile_memory (bool, optional): Track tensor memory allocations and frees.
            Ignored if ``torch_profiler_trace_dir`` and ``profiler_trace_file`` are not specified. (default: ``True``).
        torch_prof_with_stack (bool, optional): Record stack info.
            Ignored if ``torch_profiler_trace_dir`` and ``profiler_trace_file`` are not specified. (default: ``False``).
        torch_prof_with_flops (bool, optional): Estimate flops for operators.
            Ignored if ``torch_profiler_trace_dir`` and ``profiler_trace_file`` are not specified. (default: ``True``).

    Attributes:
        state (State): The :class:`.State` object used to store training state.
        evaluators (List[Evaluator]): The :class:`.Evaluator` objects to use for validation
            during training.
        logger (Logger): The :class:`.Logger` used for logging.
        engine (Engine): The :class:`.Engine` used for running callbacks and algorithms.
    """

    def __init__(
        self,
        *,
        model: ComposerModel,
        train_dataloader: Union[DataLoader, DataSpec],
        max_duration: Union[int, str, Time],
        eval_dataloader: Optional[Union[DataLoader, DataSpec, Evaluators]] = None,
        algorithms: Optional[List[Algorithm]] = None,
        optimizers: Optional[Optimizers] = None,
        schedulers: Optional[Many[ComposerScheduler]] = None,

        # device
        device: Optional[Union[str, Device]] = None,

        # training hparams
        grad_accum: int = 1,
        grad_clip_norm: Optional[float] = None,
        validate_every_n_batches: int = -1,
        validate_every_n_epochs: int = 1,
        compute_training_metrics: bool = False,
        precision: Union[str, Precision] = Precision.FP32,
        scale_schedule_ratio: float = 1.0,
        step_schedulers_every_batch: Optional[bool] = None,

        # dist hparams
        dist_timeout: float = 300.0,
        ddp_sync_strategy: Optional[Union[str, DDPSyncStrategy]] = None,

        # randomness
        seed: Optional[int] = None,
        deterministic_mode: bool = False,

        # logging and callbacks
        loggers: Optional[Sequence[LoggerCallback]] = None,
        callbacks: Sequence[Callback] = tuple(),

        # load checkpoint
        load_path_format: Optional[str] = None,
        load_object_store: Optional[ObjectStoreProvider] = None,
        load_weights_only: bool = False,
        load_strict: bool = False,
        load_chunk_size: int = 1_048_576,
        load_progress_bar: bool = True,

        # save_checkpoint
        save_folder: Optional[str] = None,
        save_name_format: str = "ep{epoch}-ba{batch}-rank{rank}",
        save_latest_format: str = "latest-rank{rank}",
        save_overwrite: bool = False,
        save_interval: Union[str, int, Time, Callable[[State, Event], bool]] = "1ep",
        save_weights_only: bool = False,

        # subset parameters
        train_subset_num_batches: Optional[int] = None,
        eval_subset_num_batches: Optional[int] = None,

        # DeepSpeed
        deepspeed_config: Union[bool, Dict[str, Any]] = False,

        # profiling
        profiler_trace_file: Optional[str] = None,
        prof_event_handlers: Sequence[ProfilerEventHandler] = tuple(),
        prof_skip_first: int = 0,
        prof_wait: int = 0,
        prof_warmup: int = 1,
        prof_active: int = 4,
        prof_repeat: int = 1,
        sys_prof_cpu: bool = True,
        sys_prof_memory: bool = False,
        sys_prof_disk: bool = False,
        sys_prof_net: bool = False,
        sys_prof_stats_thread_interval_seconds: float = 0.5,
        torch_profiler_trace_dir: Optional[str] = None,
        torch_prof_use_gzip: bool = False,
        torch_prof_record_shapes: bool = False,
        torch_prof_profile_memory: bool = True,
        torch_prof_with_stack: bool = False,
        torch_prof_with_flops: bool = True,
    ):
        # surpressing GradScaler warnings as they are always created
        # self._use_grad_scaling() will raise a RuntimeError if grad scaling is not available when it is required
        warnings.filterwarnings(action="ignore", message="torch.cuda.amp.GradScaler")

        # ScaleSchedule is a deprecated algorithm, but if it is used, updated SSR with its ratio.
        # TODO(#434): Remove this completely.
        for algorithm in algorithms or []:
            if isinstance(algorithm, ScaleSchedule):
                scale_schedule_ratio = algorithm.ratio

        if isinstance(max_duration, str):
            max_duration = Time.from_timestring(max_duration)
        elif isinstance(max_duration, int):
            max_duration = Time.from_epoch(max_duration)

        orig_max_duration = max_duration

        if scale_schedule_ratio != 1.0:
            max_duration = cast(Time[int], orig_max_duration * scale_schedule_ratio)
            log.info(f'max_duration changed from {orig_max_duration} to {max_duration}')
            if max_duration.value == 0:
                raise ValueError(
                    'Scale schedule has reduced the max_duration to 0. Set a higher ratio or use more epochs.')

        if isinstance(deepspeed_config, bool):
            self._deepspeed_config = {} if deepspeed_config else None
        else:
            self._deepspeed_config = deepspeed_config

        if not device:
            self._device = DeviceCPU() if not self.deepspeed_enabled else DeviceGPU()
        elif isinstance(device, str):
            if device == 'cpu':
                self._device = DeviceCPU()
            elif device == 'gpu':
                self._device = DeviceGPU()
            else:
                raise ValueError(f'device ({device}) must be one of (cpu, gpu).')
        else:
            if not isinstance(device, Device):
                raise ValueError('device must be of class Device')
            self._device = device

        if self.deepspeed_enabled or dist.get_world_size() > 1:
            # deepspeed requires torch.distributed to be initialized, even if the world size is 1
            # distributed is always required with multi-rank training
            dist.initialize_dist(self._device.dist_backend, datetime.timedelta(seconds=dist_timeout))

        if not seed:
            seed = reproducibility.get_random_seed()

        # Ensure that each process has a seed = rank_zero_seed + global_rank
        # This "deterministically different" seed behavior is required to be able
        # to restore seeds when resuming form checkpoints, since only the
        # `rank_zero_seed` is stored on state.
        if seed < 0 or seed > reproducibility.MAX_SEED:
            raise ValueError(f"Invalid seed: {seed}. It must be on [0; 2**32 - 1)")
        rank_zero_seed = self._device.tensor_to_device(torch.tensor(
            [seed], dtype=torch.int64))  # using int64 to prevent overflow
        dist.broadcast(rank_zero_seed, src=0)
        rank_zero_seed = rank_zero_seed.item()
        assert isinstance(rank_zero_seed, int)
        seed = rank_zero_seed + dist.get_global_rank()
        log.info(f"Setting seed to {seed}")

        # If hparams is used to create the Trainer this function is called twice
        # which is okay because all runs with the hparams codepath will do this
        reproducibility.seed_all(seed)

        if not algorithms:
            algorithms = []

        # some algorithms require specific settings
        self._backwards_create_graph = any(map(lambda x: x.backwards_create_graph, algorithms))
        find_unused_parameters = any(map(lambda x: x.find_unused_parameters, algorithms))
        self._find_unused_parameters = find_unused_parameters

        if ddp_sync_strategy is None:
            self._ddp_sync_strategy = DDPSyncStrategy.SINGLE_AUTO_SYNC if not find_unused_parameters else DDPSyncStrategy.FORCED_SYNC
        else:
            self._ddp_sync_strategy = DDPSyncStrategy(ddp_sync_strategy)

        # convert eval_dataloader to `List[Evaluator]`
        self.evaluators: List[Evaluator] = []
        for evaluator in ensure_tuple(eval_dataloader):
            if isinstance(evaluator, Evaluator):
                self.evaluators.append(evaluator)
            else:
                metrics = model.metrics(train=False)
                evaluator = Evaluator(label="eval_dataset", dataloader=evaluator, metrics=metrics)
                self.evaluators.append(evaluator)

        self._eval_subset_num_batches = eval_subset_num_batches

        # do a check here to make sure there is at least one validation set
        if len(self.evaluators) == 0:
            warnings.warn(
                textwrap.dedent("""No evaluation dataset was specified. Please specify `eval_dataloader` to periodically
                evaluate your model while training."""),
                category=UserWarning)

        if not isinstance(train_dataloader, DataSpec):
            train_dataloader = DataSpec(train_dataloader)

        self._train_data_spec = train_dataloader
        unwrapped_data_loader = unwrap_data_loader(self._train_data_spec.dataloader)
        if isinstance(unwrapped_data_loader, torch.utils.data.DataLoader):
            if unwrapped_data_loader._iterator is not None:
                raise ValueError(
                    textwrap.dedent("""\
                    The `train_dataloader` has an active iterator. This could occur
                    if `persistent_workers=True` and the dataloader has already been iterated,
                    or if the dataloader is mid-epoch. It is required that the training dataloader
                    does not have an active iterator, so CPU dataset augmentations can be
                    correctly inserted.

                    To fix, please do not iterate over the dataloader before passing it into
                    the trainer."""))

        # TODO(#123): DeepSpeed still needs a precision context, but it's not completely clear how to
        # handle this with our version of Pytorch
        precision_context = self._device.precision_context if not self.deepspeed_enabled else cast(
            Callable[..., ContextManager], contextlib.nullcontext)
        if isinstance(precision, str):
            precision = Precision(precision)

        # optimizers and schedulers
        if not optimizers:
            optimizers = DecoupledSGDW(list(model.parameters()), lr=0.1)
            warnings.warn(f"No optimizer was specified. Defaulting to {repr(optimizers)}")

        num_optimizers = len(ensure_tuple(optimizers))
        if num_optimizers != 1:
            raise NotImplementedError(f"Only one optimizer is supported; found {num_optimizers} optimizers")

        self.state = State(
            max_duration=max_duration,
            rank_zero_seed=rank_zero_seed,
            algorithms=algorithms,
            model=model,
            callbacks=callbacks,
            grad_accum=grad_accum,
            precision=precision,
            precision_context=precision_context,
            train_dataloader=train_dataloader.dataloader,
            evaluators=self.evaluators,
            optimizers=optimizers,
            steps_per_epoch=train_subset_num_batches,
        )

        pytorch_schedulers = [
            scheduler for scheduler in ensure_tuple(schedulers) if isinstance(scheduler, PyTorchScheduler)
        ]
        if len(pytorch_schedulers) > 0:
            if step_schedulers_every_batch is True:
                log.info(
                    textwrap.dedent(f"""\
                    Schedulers are being steped every batch, as `step_schedulers_every_batch` is True.
                    The trainer cannot automatically convert the parameters (e.g. step_size, T_max) of the
                    PyTorch {type(pytorch_schedulers[0]).__name__} scheduler to be in terms of batches.
                    Please ensure that the scheduler parameters are in terms of batches, not epochs.
                    Alternatively, use a ComposerScheduler. For more information, see
                    https://docs.mosaicml.com/en/v{composer.__version__}/trainer/schedulers.html.
                    """))

            if step_schedulers_every_batch is None:
                # only if all schedulers are ComposerSchedulers, then we can step every batch by default
                step_schedulers_every_batch = False

            log.info(
                textwrap.dedent(f"""\
                Schedulers will be stepped every epoch because the Trainer was constructed with a PyTorch
                {type(pytorch_schedulers[0]).__name__} scheduler. To step the schedulers every batch, adjust the
                scheduler parameters (e.g. step_size, T_max) to be in terms of batches and set
                `step_schedulers_every_batch` to True, or alternatively use a ComposerScheduler. For more information,
                see https://docs.mosaicml.com/en/v{composer.__version__}/trainer/schedulers.html."""))

        else:
            step_schedulers_every_batch = True

        self._step_schedulers_every_batch = step_schedulers_every_batch

        for scheduler in ensure_tuple(schedulers):
            if isinstance(scheduler, PyTorchScheduler):
                scale_pytorch_scheduler(scheduler, scale_schedule_ratio)
                self.state.schedulers.append(scheduler)
            else:  # it's a composer scheduler
                self.state.schedulers.append(compile_composer_scheduler(scheduler, self.state, scale_schedule_ratio))

        if len(self.state.schedulers) == 0:
            warnings.warn(f"NoSchedulerWarning: No schedulers were specified. The learning rate will be constant.")

        # Configure profilers if profiling is enabled
        if profiler_trace_file:
            self.state.profiler = Profiler(state=self.state,
                                           event_handlers=prof_event_handlers,
                                           skip_first=prof_skip_first,
                                           wait=prof_wait,
                                           warmup=prof_warmup,
                                           active=prof_active,
                                           repeat=prof_repeat,
                                           merged_trace_file=profiler_trace_file)
            self.state.callbacks.extend(self.state.profiler.event_handlers)

            self.state.callbacks.append(DataLoaderProfiler())

            if sys_prof_cpu or sys_prof_memory or sys_prof_disk or sys_prof_net:
                self.state.callbacks.append(
                    SystemProfiler(profile_cpu=sys_prof_cpu,
                                   profile_memory=sys_prof_memory,
                                   profile_disk=sys_prof_disk,
                                   profile_net=sys_prof_net,
                                   stats_thread_interval_seconds=sys_prof_stats_thread_interval_seconds))

            if torch_profiler_trace_dir:
                self.state.callbacks.append(
                    TorchProfiler(tensorboard_trace_handler_dir=torch_profiler_trace_dir,
                                  tensorboard_use_gzip=torch_prof_use_gzip,
                                  record_shapes=torch_prof_record_shapes,
                                  profile_memory=torch_prof_profile_memory,
                                  with_stack=torch_prof_with_stack,
                                  with_flops=torch_prof_with_flops))

        if loggers is None:
            loggers = [TQDMLogger()]
        self.logger = Logger(self.state, loggers)
        self.state.callbacks = list(cast(List[Callback], loggers)) + self.state.callbacks

        self._checkpoint_saver = None
        if save_folder is not None:
            self._checkpoint_saver = CheckpointSaver(
                save_folder=save_folder,
                name_format=save_name_format,
                save_latest_format=save_latest_format,
                overwrite=save_overwrite,
                weights_only=save_weights_only,
                save_interval=save_interval,
            )
            self.state.callbacks.append(self._checkpoint_saver)

        self.engine = Engine(
            state=self.state,
            logger=self.logger,
        )

        self._validate_every_n_batches = validate_every_n_batches
        self._validate_every_n_epochs = validate_every_n_epochs
        self._compute_training_metrics = compute_training_metrics
        self._grad_clip_norm = grad_clip_norm

        if deterministic_mode:
            reproducibility.configure_deterministic_mode()

        self.engine.run_event(Event.INIT)

        assert isinstance(self.state.model, ComposerModel)
        self._original_model = self.state.model  # TODO(ravi) -- update the state to add an original model helper

        # place the state, model in the proper devices, and initialize from a checkpoint if provided
        if self.deepspeed_enabled:
            try:
                import deepspeed
            except ImportError as e:
                raise ImportError(
                    textwrap.dedent("""\
                    Composer was installed without DeepSpeed support. To use DeepSpeed with Composer, run `pip install mosaicml[deepspeed]`
                    if using pip or `pip install deepspeed>=0.5.5` if using Anaconda.""")) from e
            assert self._deepspeed_config is not None
            self._deepspeed_config = _parse_deepspeed_config(self._deepspeed_config,
                                                             state=self.state,
                                                             grad_clip_norm=self._grad_clip_norm)
            optimizer = ensure_tuple(self.state.optimizers)[0]
            (self.state.model, self.state.optimizers, _, _) = deepspeed.initialize(
                config=self._deepspeed_config,
                model=self.state.model,
                optimizer=optimizer,
            )
            # The deepspeed engine is responsible for serializing the model and optimizer state,
            # so these attributes should not be serialized with the composer state.
            if "model" in self.state.serialized_attributes:
                self.state.serialized_attributes.remove("model")

            if "optimizers" in self.state.serialized_attributes:
                self.state.serialized_attributes.remove("optimizers")

        # If using DeepSpeed, the model must be loaded from checkpoint after the engine has been
        # initialized, but if using PyTorch DDP, the model must be loaded before it is wrapped with
        # DDP.

        self._rng_state = None
        if load_path_format is not None:
            self._rng_state = load_checkpoint(state=self.state,
                                              path_format=load_path_format,
                                              object_store=load_object_store,
                                              load_weights_only=load_weights_only,
                                              strict_model_weights=load_strict,
                                              chunk_size=load_chunk_size,
                                              progress_bar=load_progress_bar)
            reproducibility.seed_all(self.state.seed)

        if not self.deepspeed_enabled:
            host_model_params = self.state.model.parameters()
            self.state.model = self._device.module_to_device(self.state.model)
            device_model_params = self.state.model.parameters()

            # use surgery to update the parameters of the optimizers, now that the model is on the device
            # see https://pytorch.org/docs/stable/optim.html#constructing-it
            module_surgery.replace_params_in_optimizer(old_params=host_model_params,
                                                       new_params=device_model_params,
                                                       optimizers=self.state.optimizers)

            # Move any remaining optimizer parameters onto the device
            self.state.optimizers = map_collection(self.state.optimizers, self._device.optimizer_to_device)

            if dist.get_world_size() > 1:
                # Only wrap the module if required
                self.state.model = _prepare_ddp_module(self.state.model, self._find_unused_parameters)

    @property
    def deepspeed_enabled(self) -> bool:
        """``True`` if DeepSpeed is being used for training and ``False`` otherwise.

        .. seealso:: `DeepSpeed's documentation <https://www.deepspeed.ai/docs/config-json/>`_
        """
        return self._deepspeed_config is not None

    @property
    def saved_checkpoints(self) -> Dict[Timestamp, List[str]]:
        """The times and paths to checkpoint files saved across all ranks during training.

        Returns:

            Dict[Timestamp, List[str]]: A dictionary mapping a save :class:`.Timestamp`. to a list of
                filepaths, indexed by global rank, corresponding to the checkpoints saved at that time.

        .. note::

            When using DeepSpeed, the index of a filepath corresponds to the
            global rank of the process that wrote that file. These filepaths are valid only on
            the global rank's node. Otherwise, when not using DeepSpeed, this list will contain
            only one filepath since only rank zero saves checkpoints.
        """
        if self._checkpoint_saver is None:
            raise RuntimeError("`save_folder` was not specified, so no checkpoints were saved.")
        return self._checkpoint_saver.saved_checkpoints

    @property
    def checkpoint_folder(self) -> Optional[str]:
        """The folder in which checkpoints are stored.

        Returns:
            Optional[str]: The checkpoint folder, or None, if checkpoints were not saved.

            If an absolute path was specified for ``save_folder`` upon trainer instantiation, then that path will be
            used. Otherwise, this folder is relative to the :mod:`~composer.utils.run_directory` of the training run
            (e.g. ``{run_directory}/{save_folder}``). If no run directory is provided, then by default, it is of the
            form ``runs/<timestamp>/rank_<GLOBAL_RANK>/<save_folder>`` where ``timestamp`` is the start time of the
            run in iso-format, ``GLOBAL_RANK`` is the global rank of the process, and ``save_folder`` is the
            ``save_folder`` argument provided upon construction.
        """
        if self._checkpoint_saver is None:
            raise RuntimeError("`save_folder` was not specified, so no checkpoints were saved.")
        return self._checkpoint_saver.checkpoint_folder

    def fit(self):
        """Train and evaluate the model on the provided data."""
        try:
            self._train_loop()
        finally:
            self.engine.close()

    def _ensure_metrics_device_and_dtype(self, metrics: MetricCollection):
        # Safety check to ensure the metric and data are on the same device. Normally not
        # needed because the metric is automatically on the same device as the model.
        # See https://torchmetrics.readthedocs.io/en/latest/pages/overview.html for details.
        metrics = self._device.module_to_device(metrics)

        # HACK: DeepSpeed somehow manages to convert metric internal states to its own dtype. When
        # running with FP16, this tends to result in overflows. Let's assume FP32 is good enough.
        for _, metric in metrics.items():
            metric.set_dtype(torch.float32)  # type: ignore

        return metrics

    def _compute_and_log_metrics(self, metrics: Metrics, *, is_train: bool, is_batch: bool, logging_label: str = ''):
        """Computes metrics, logs the results, and resets the metrics.

        Args:
            metrics (Metrics): The metrics to compute.
            is_train (bool): True for training metrics, False for evaluation metrics.
            is_batch (bool): True if logging at batch level, false for epoch level.
            logging_label (str): Should be left as empty string if called for training metrics.
                Should be the evaluator label if called on evaluator metrics.
        """
        computed_metrics = metrics.compute()
        for name, value in computed_metrics.items():
            log_level = LogLevel.BATCH if is_batch else LogLevel.EPOCH
            suffix = 'train' if is_train else 'val'

            # default label given to evaluator created by val_dataset parameter
            if not logging_label or logging_label == "eval_dataset":
                label = f'{name.lower()}/{suffix}'
            else:
                label = f'{logging_label}/{name.lower()}_{suffix}'
            self.logger.metric(log_level, {label: value})
        metrics.reset()

    def _spin_dataloaders(self):
        """Spin the dataloaders to restore sampler state.

        Only one batch must be loaded to seed the sampler's generator. since only the first batch is being loaded, the
        dataloader may not be completely iterated through.
        """
        # spin the evaluator dataloaders once to initialize its sampler deterministically
        # so it does not affect any other RNG reads
        for evaluator in self.state.evaluators:
            dataloader = evaluator.dataloader.dataloader
            if isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
                dataloader.sampler.set_epoch(0)
            for _ in dataloader:
                break

        # spin the train dataloader's sampler to get to the state of the desired epoch
        for epoch in range(int(self.state.timer.epoch)):
            if isinstance(self.state.train_dataloader.sampler, torch.utils.data.DistributedSampler):
                self.state.train_dataloader.sampler.set_epoch(epoch)
            for _ in self.state.train_dataloader:
                break

    def _train_loop(self) -> None:
        """Run training for the specified number of epochs and log results."""
        # print training start
        self.logger.metric_fit({"trainer/algorithms": [str(algo) for algo in self.state.algorithms]})

        if self._compute_training_metrics:
            log.warn('Computing model evaluation metrics during training.'
                     ' This doubles the number of forward passes and may lead'
                     ' to a throughput degradation.')
            train_metrics = self._original_model.metrics(train=True)
            if isinstance(train_metrics, Metric):
                # Forcing metrics to be a MetricCollection simplifies logging results
                train_metrics = MetricCollection([train_metrics])

            train_metrics = self._ensure_metrics_device_and_dtype(train_metrics)
        else:
            train_metrics = None

        self.engine.run_event(Event.FIT_START)

        self.state.scaler = ClosureGradScaler() if self._use_closures() else GradScaler()
        use_grad_scaling = self._use_grad_scaling(self.state.precision, self.state.scaler)

        self._spin_dataloaders()

        if self.state.timer.batch_in_epoch == 0 and self._rng_state is not None:
            # only restore the rng state here if the step in the current epoch is zero.
            reproducibility.load_rng_state(self._rng_state)
            self._rng_state = None

        while self.state.timer < self.state.max_duration:
            try:
                self.state.model.train()

                if int(self.state.timer.batch_in_epoch) == 0:
                    self.engine.run_event(Event.EPOCH_START)
                    self.logger.metric_epoch({"epoch": int(self.state.timer.epoch)})

                if isinstance(self.state.train_dataloader.sampler, torch.utils.data.DistributedSampler):
                    self.state.train_dataloader.sampler.set_epoch(int(self.state.timer.epoch))

                for batch_idx, self.state.batch in enumerate(
                        itertools.islice(self.state.train_dataloader, self.state.steps_per_epoch)):

                    # if resuming, skip dataloader forward to the minibatch index
                    if batch_idx < int(self.state.timer.batch_in_epoch):
                        # Restore the RNG state immediately before the next batch is yielded from the dataloader
                        if batch_idx + 1 == int(self.state.timer.batch_in_epoch) and self._rng_state is not None:
                            reproducibility.load_rng_state(self._rng_state)
                            self._rng_state = None
                        continue

                    self.state.batch = self._device.batch_to_device(self.state.batch)
                    self.state.batch = self._train_data_spec.device_transforms(self.state.batch)
                    self.state.batch_num_samples = self._train_data_spec.get_num_samples_in_batch(self.state.batch)
                    self.state.batch_num_tokens = self._train_data_spec.get_num_tokens_in_batch(self.state.batch)

                    if self.deepspeed_enabled:
                        self.state.batch = _fix_batch_precision_for_deepspeed(self.state.batch, self.state.precision)

                    if self._compute_training_metrics:
                        # compute metrics on the training set
                        assert train_metrics is not None
                        self.state.model.eval()
                        with torch.no_grad():
                            for eval_microbatch in self._train_data_spec.split_batch(
                                    self.state.batch, self.state.grad_accum):
                                # TODO: Detect if self.run_event(Event.AFTER_DATALOADER) changes the training
                                # data and if so print a warning that metrics may return unexpected results
                                outputs, targets = self._original_model.validate(eval_microbatch)
                                train_metrics.update(outputs, targets)

                    self.state.model.train()

                    self.engine.run_event(Event.AFTER_DATALOADER)

                    num_samples_in_batch = self._device.tensor_to_device(
                        torch.tensor([self.state.batch_num_samples], dtype=torch.int))
                    num_tokens_in_batch = self._device.tensor_to_device(
                        torch.tensor([self.state.batch_num_tokens], dtype=torch.int))
                    dist.all_reduce(num_samples_in_batch, reduce_operation="SUM")
                    dist.all_reduce(num_tokens_in_batch, reduce_operation="SUM")

                    self.engine.run_event(Event.BATCH_START)
                    self.logger.metric_batch({
                        "trainer/global_step": int(self.state.timer.batch),
                        "trainer/batch_idx": self.state.timer.batch_in_epoch.value,
                    })
                    total_loss = None
                    microbatches = self._train_data_spec.split_batch(self.state.batch, self.state.grad_accum)
                    if self.deepspeed_enabled:
                        total_loss = self._train_batch(microbatches)
                    elif self._use_closures():
                        for optimizer in self.state.optimizers:
                            if use_grad_scaling:
                                total_loss = self.state.scaler.step(
                                    optimizer, closure=lambda **kwargs: self._train_batch(microbatches, **kwargs))
                            else:
                                total_loss = optimizer.step(
                                    closure=lambda **kwargs: self._train_batch(microbatches, **kwargs).item())
                    else:
                        total_loss = self._train_batch(microbatches)
                        for optimizer in self.state.optimizers:
                            if use_grad_scaling:
                                self.state.scaler.step(optimizer)
                            else:
                                optimizer.step()

                    if use_grad_scaling:
                        self.state.scaler.update()

                    if total_loss is not None:
                        if not isinstance(total_loss, torch.Tensor):
                            total_loss = self._device.tensor_to_device(torch.tensor([total_loss]))

                        # total_loss can be None if gradient scaling failed
                        dist.all_reduce(total_loss, reduce_operation="SUM")
                        full_loss = total_loss.cpu().item()
                        self.logger.metric_batch({'loss/train': full_loss / dist.get_world_size()})

                    if self._compute_training_metrics:
                        assert train_metrics is not None
                        self._compute_and_log_metrics(train_metrics, is_train=True, is_batch=True)

                    self.state.timer.on_batch_complete(
                        samples=int(num_samples_in_batch.item()),
                        tokens=int(num_tokens_in_batch.item()),
                    )

                    if self._step_schedulers_every_batch:
                        for scheduler in self.state.schedulers:
                            scheduler.step()

                    self.engine.run_event(Event.BATCH_END)

                    if self._validate_every_n_batches > 0 and int(
                            self.state.timer.batch) % self._validate_every_n_batches == 0:
                        self.eval(is_batch=True)

                    self.engine.run_event(Event.BATCH_CHECKPOINT)

                    if self.state.timer >= self.state.max_duration:
                        # If max_duration is specified in batches, samples, or tokens, and
                        # and the max_duration is reached mid-epoch, then break out of the dataloader
                        # to finish the epoch early and finish training.
                        break

            except BreakEpochException:
                log.info(f'Skipping the rest of Epoch {int(self.state.timer.epoch)}')

            self.state.timer.on_epoch_complete()

            if not self._step_schedulers_every_batch:
                for scheduler in self.state.schedulers:
                    scheduler.step()

            self.engine.run_event(Event.EPOCH_END)

            if self._validate_every_n_epochs > 0 and int(self.state.timer.epoch) % self._validate_every_n_epochs == 0:
                self.eval(is_batch=False)

            self.engine.run_event(Event.EPOCH_CHECKPOINT)

    def _train_batch(self, microbatches: Sequence[Batch], ddp_sync: bool = True):
        """Run training on a full batch of data.

        Args:
            microbatches (Sequence[Batch]): The microbatches which make up the batch.
            ddp_sync (bool): True to sync gradients between devices on every backwards
                pass and False to only sync gradients after each device has finished
                computing a gradient on it's entire set of microbatches. (default: ``True``)
        """
        if ddp_sync or not isinstance(self.state.model, DistributedDataParallel):
            context = contextlib.nullcontext
        else:
            context = cast(Callable[[], ContextManager], self.state.model.no_sync)

        with context():
            return self._train_batch_inner(microbatches)

    def _train_batch_inner(self, microbatches: Sequence[Batch]):
        """Iterate over microbatches and compute the loss that will be used to step the optimizer."""
        self.engine.run_event(Event.BEFORE_TRAIN_BATCH)

        assert self.state.optimizers is not None
        assert self.state.scaler is not None

        use_grad_scaling = self._use_grad_scaling(self.state.precision, self.state.scaler)

        if not self.deepspeed_enabled:
            for optimizer in self.state.optimizers:
                optimizer.zero_grad()

        # tracker for gradient accumulation
        total_loss = self._device.tensor_to_device(torch.zeros(size=(1,)))
        current_batch_size = sum([self._train_data_spec.get_num_samples_in_batch(batch) for batch in microbatches])

        for microbatch_idx, self.state.batch in enumerate(microbatches):
            self.state.batch_num_tokens = self._train_data_spec.get_num_tokens_in_batch(self.state.batch)
            self.state.batch_num_samples = self._train_data_spec.get_num_samples_in_batch(self.state.batch)
            is_final_microbatch = microbatch_idx + 1 == len(microbatches)
            sync_context = contextlib.nullcontext() if self.deepspeed_enabled else _ddp_sync_context(
                self.state, is_final_microbatch, self._ddp_sync_strategy)
            with sync_context:
                # forward pass
                self.engine.run_event(Event.BEFORE_FORWARD)

                with self.state.precision_context:
                    self.state.outputs = self.state.model.forward(self.state.batch)

                self.engine.run_event(Event.AFTER_FORWARD)

                # loss
                self.engine.run_event(Event.BEFORE_LOSS)

                with self.state.precision_context:
                    self.state.loss = self._original_model.loss(self.state.outputs, self.state.batch)

                # We always want to scale loss by the grad_accum before the backwards pass and
                # also for sake of metrics. Complicating matters, the DeepSpeed engine does its
                # own scaling when we call `.backward`, but this isn't in place so we still need
                # to scale for sake of metrics after the `.backward` call.

                # Loss is added to losses with clone to not scale the loss for the step printout
                # Likely need to look into the performance impact
                if not self.deepspeed_enabled:
                    for loss in ensure_tuple(self.state.loss):
                        loss.mul_(self.state.batch_num_samples / current_batch_size)
                        total_loss += loss.detach().clone()

                assert self.state.loss is not None
                self.engine.run_event(Event.AFTER_LOSS)

                # backward
                self.engine.run_event(Event.BEFORE_BACKWARD)

                if use_grad_scaling:
                    self.state.loss = cast(torch.Tensor, self.state.scaler.scale(self.state.loss))

                if self.deepspeed_enabled:
                    self.state.deepspeed_model.backward(self.state.loss)

                    # This is the same loss scaling and reporting we skipped earlier.
                    for loss in ensure_tuple(self.state.loss):
                        loss.mul_(self.state.batch_num_samples / current_batch_size)
                        total_loss += loss.detach().clone()
                else:
                    for loss in ensure_tuple(self.state.loss):
                        loss.backward(create_graph=self._backwards_create_graph)

                self.engine.run_event(Event.AFTER_BACKWARD)

            if self.deepspeed_enabled:
                self.state.deepspeed_model.step()

        # Unscale gradients before `Event.AFTER_TRAIN_BATCH`
        if use_grad_scaling:
            for optimizer in ensure_tuple(self.state.optimizers):
                self.state.scaler.unscale_(optimizer)

        # clip gradients if the magnitude is too large
        if not self.deepspeed_enabled and self._grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.state.model.parameters(),
                max_norm=self._grad_clip_norm,
            )

        self.engine.run_event(Event.AFTER_TRAIN_BATCH)

        return total_loss

    def eval(self, is_batch: bool):
        """Evaluate the model on the provided evaluation data and log appropriate metrics.

        Args:
            is_batch (bool): True to log metrics with ``LogLevel.BATCH``
                and False to log metrics with ``LogLevel.EPOCH``.
        """
        restore_model_train = self.state.model.training

        self.state.model.eval()
        with torch.no_grad():

            self.engine.run_event(Event.EVAL_START)

            for evaluator in self.state.evaluators:
                dataloader = evaluator.dataloader.dataloader
                metrics = self._ensure_metrics_device_and_dtype(evaluator.metrics)
                if isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
                    # The distributed sampler uses `set_epoch` to set the random seed
                    # Because evaluation can run on each batch, we use the batch to seed the sampler
                    # so each evaluation will get a proper shuffle.
                    # The epoch provided to `set_epoch` need not be sequential, so this is fine.
                    dataloader.sampler.set_epoch(int(self.state.timer.batch))

                for self.state.batch in itertools.islice(dataloader, self._eval_subset_num_batches):
                    self.state.batch = self._device.batch_to_device(self.state.batch)
                    if evaluator.dataloader.device_transforms:
                        self.state.batch = evaluator.dataloader.device_transforms(self.state.batch)
                    self.state.batch_num_samples = evaluator.dataloader.get_num_samples_in_batch(self.state.batch)
                    self.state.batch_num_tokens = evaluator.dataloader.get_num_tokens_in_batch(self.state.batch)

                    if self.deepspeed_enabled:
                        self.state.batch = _fix_batch_precision_for_deepspeed(self.state.batch, self.state.precision)

                    self.engine.run_event(Event.EVAL_BATCH_START)

                    self.engine.run_event(Event.EVAL_BEFORE_FORWARD)
                    self.state.outputs, targets = self._original_model.validate(self.state.batch)
                    self.engine.run_event(Event.EVAL_AFTER_FORWARD)

                    metrics.update(self.state.outputs, targets)

                    self.engine.run_event(Event.EVAL_BATCH_END)

                self.logger.metric_epoch({"epoch": self.state.timer.epoch.value})
                self.logger.metric_batch({"trainer/global_step": self.state.timer.batch.value})

                self._compute_and_log_metrics(metrics, is_train=False, is_batch=is_batch, logging_label=evaluator.label)

            self.engine.run_event(Event.EVAL_END)

        if restore_model_train:
            self.state.model.train()

    def _use_grad_scaling(self, precision: Union[str, Precision], scaler: Optional[GradScaler]) -> bool:
        """Determines based on precision when to use grad scaling.

        By default, the pytorch GradScaler is a no-op if running on
        unsupported hardware. Here we raise a RuntimeError instead.

        Args:
            precision (Precision): Numerical precision, based on the Precision Enum.
            scaler (GradScaler): Used to make sure that the scaler is enabled when
            using grad scaling.

        Raises:
            RuntimeError:
                Occurs when attempting to use grad scaling without the scaler
                enabled. Likely due to hardware not supporting the provided precision.
        """
        if self.deepspeed_enabled:
            return False

        precision = Precision(precision)
        use_grad_scaling = precision == Precision.AMP

        if use_grad_scaling and (scaler is None or not scaler.is_enabled()):
            raise RuntimeError(f'Attempting to use grad scaling with {precision}, but scaler is not enabled.'
                               f'Potentially your hardware does not support Precision {precision}.')
        return use_grad_scaling

    def _use_closures(self) -> bool:
        """Determines based on precision and optimizers whether to use closures.

        We default to using closures unless AMP is enabled, in which case we only allow closures when using optimizers
        with the _step_supports_amp_closure flag.
        """
        if self.deepspeed_enabled:
            return False

        if self.state.precision != Precision.AMP:
            return True

        if self.state.optimizers is None:
            raise RuntimeError("state.optimizers must be set before `_use_closures` can be determined")

        return all(
            getattr(optimizer, "_step_supports_amp_closure", False)
            for optimizer in ensure_tuple(self.state.optimizers))

    def save_checkpoint(self, name_format: str = "ep{epoch}-ba{batch}-rank{rank}", *, weights_only: bool = False):
        """Checkpoint the training :class:`~.State`.

        Args:
            name_format (str, optional): See :func:`.save_checkpoint`.
            weights_only (bool, optional): See :func:`.save_checkpoint`.

        Returns:
            List[pathlib.Path]: See :func:`.save_checkpoint`.
        """
        return save_checkpoint(state=self.state, name_format=name_format, weights_only=weights_only)
