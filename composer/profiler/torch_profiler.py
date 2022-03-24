# Copyright 2021 MosaicML. All Rights Reserved.

"""Profiler to collect :mod:`torch` performance metrics during training."""

from __future__ import annotations

import json
import os
from typing import Optional, OrderedDict

import torch.profiler
from torch.profiler.profiler import ProfilerAction as TorchProfilerAction

from composer.core import Callback, State
from composer.loggers import Logger
from composer.loggers.logger import LogLevel
from composer.profiler._profiler_action import ProfilerAction
from composer.utils import dist, ensure_folder_is_empty, format_name_with_dist, format_name_with_dist_and_time

__all__ = ["TorchProfiler"]


class TorchProfiler(Callback):
    """Profile the execution using the :class:`PyTorch Profiler <torch.profiler.profile>`.

    Profiling results are stored in TensorBoard format in the directory specified by ``folder_format``.

    .. note::

        The Composer :class:`~composer.trainer.trainer.Trainer` creates an instance of :class:`.TorchProfiler` when
        any of the PyTorch Profiler arguments (``torch_prof_record_shapes``, ``torch_prof_profile_memory``,
        ``torch_prof_with_stack``, or ``torch_prof_with_flops``) are enabled.

        When using the Composer :class:`~composer.trainer.trainer.Trainer`, one does not need to directly create an
        instance of the :class:`.TorchProfiler` callback.


    To view profiling results, run::

        pip install tensorbaord torch_tb_profiler
        tensorboard --logdir path/to/torch/trace_folder

    .. note::

        See :doc:`profiler` for additional usage details on the :class:`torch.profiler.profile`.

    .. note::

        Enabling shape and stack tracing results in additional overhead.
        When ``record_shapes=True`` is specified, the profiler will temporarily hold references to tensors which
        may prevent certain optimizations that depend on the reference count and can introduce extra tensor copies.

    Args:
        folder_format (str, optional): Format string for the folder containing the Torch Profiler trace files.
            Defaults to ``'{run_name}/torch_traces'``.

            The following format variables are available:

            +------------------------+-------------------------------------------------------+
            | Variable               | Description                                           |
            +========================+=======================================================+
            | ``{run_name}``         | The name of the training run. See                     |
            |                        | :attr:`~composer.core.logging.Logger.run_name`.       |
            +------------------------+-------------------------------------------------------+
            | ``{rank}``             | The global rank, as returned by                       |
            |                        | :func:`~composer.utils.dist.get_global_rank`.         |
            +------------------------+-------------------------------------------------------+
            | ``{local_rank}``       | The local rank of the process, as returned by         |
            |                        | :func:`~composer.utils.dist.get_local_rank`.          |
            +------------------------+-------------------------------------------------------+
            | ``{world_size}``       | The world size, as returned by                        |
            |                        | :func:`~composer.utils.dist.get_world_size`.          |
            +------------------------+-------------------------------------------------------+
            | ``{local_world_size}`` | The local world size, as returned by                  |
            |                        | :func:`~composer.utils.dist.get_local_world_size`.    |
            +------------------------+-------------------------------------------------------+
            | ``{node_rank}``        | The node rank, as returned by                         |
            |                        | :func:`~composer.utils.dist.get_node_rank`.           |
            +------------------------+-------------------------------------------------------+

            For example, if the ``run_name`` is ``'awesome_training_run'``, and the default ``folder_format`` of
            '{run_name}/torch_traces' is used, Torch Profiler traces will be stored in
            ``'awesome_training_run/torch_traces'``.

        filename_format (str, optional): A format string describing how to name Torch Profiler trace files.
            Defaults to ``'ep{epoch}-ba{batch}-rank{rank}.json'``.

            At the end of each batch where :meth:`~composer.profiler.Profiler.get_action` returns
            :attr:`~composer.profiler._profiler_action.ProfilerAction.ACTIVE_AND_SAVE`, trace files are saved
            approximately to ``{folder_format.format(...)}/{filename_format.format(...)}``.

            The following format variables are available:

            +------------------------+-------------------------------------------------------+
            | Variable               | Description                                           |
            +========================+=======================================================+
            | ``{run_name}``         | The name of the training run. See                     |
            |                        | :attr:`~composer.core.logging.Logger.run_name`.       |
            +------------------------+-------------------------------------------------------+
            | ``{rank}``             | The global rank, as returned by                       |
            |                        | :func:`~.dist.get_global_rank`.                       |
            +------------------------+-------------------------------------------------------+
            | ``{local_rank}``       | The local rank of the process, as returned by         |
            |                        | :func:`~.dist.get_local_rank`.                        |
            +------------------------+-------------------------------------------------------+
            | ``{world_size}``       | The world size, as returned by                        |
            |                        | :func:`~.dist.get_world_size`.                        |
            +------------------------+-------------------------------------------------------+
            | ``{local_world_size}`` | The local world size, as returned by                  |
            |                        | :func:`~.dist.get_local_world_size`.                  |
            +------------------------+-------------------------------------------------------+
            | ``{node_rank}``        | The node rank, as returned by                         |
            |                        | :func:`~.dist.get_node_rank`.                         |
            +------------------------+-------------------------------------------------------+
            | ``{epoch}``            | The total epoch count, as returned by                 |
            |                        | :meth:`~composer.core.time.Timer.epoch`.              |
            +------------------------+-------------------------------------------------------+
            | ``{batch}``            | The total batch count, as returned by                 |
            |                        | :meth:`~composer.core.time.Timer.batch`.              |
            +------------------------+-------------------------------------------------------+
            | ``{batch_in_epoch}``   | The batch count in the current epoch, as returned by  |
            |                        | :meth:`~composer.core.time.Timer.batch_in_epoch`.     |
            +------------------------+-------------------------------------------------------+
            | ``{sample}``           | The total sample count, as returned by                |
            |                        | :meth:`~composer.core.time.Timer.sample`.             |
            +------------------------+-------------------------------------------------------+
            | ``{sample_in_epoch}``  | The sample count in the current epoch, as returned by |
            |                        | :meth:`~composer.core.time.Timer.sample_in_epoch`.    |
            +------------------------+-------------------------------------------------------+
            | ``{token}``            | The total token count, as returned by                 |
            |                        | :meth:`~composer.core.time.Timer.token`.              |
            +------------------------+-------------------------------------------------------+
            | ``{token_in_epoch}``   | The token count in the current epoch, as returned by  |
            |                        | :meth:`~composer.core.time.Timer.token_in_epoch`.     |
            +------------------------+-------------------------------------------------------+

            Consider the following scenario, where:

            *   The :attr:`~.Logger.run_name` is 'awesome-training-run'
            *   The default ``trace_folder_format='{run_name}/torch_traces'`` is used.
            *   The default ``name_format='ep{epoch}-ba{batch}-rank{rank}.json'`` is used.
            *   The current epoch count is ``1``.
            *   The current batch count is ``42``.

            Each rank (process) will save traces to::

                awesome-training-run/torch_traces/ep1-ba42-rank0.json
                awesome-training-run/torch_traces/ep1-ba42-rank1.json
                awesome-training-run/torch_traces/ep1-ba42-rank2.json
                ...

        artifact_name_format (str, optional): Format string for a Torch Profiler trace file's artifact name.
            Defaults to ``'{run_name}/torch_traces/ep{epoch}-ba{batch}-rank{rank}.json'``.

            Whenever a trace file is saved, it is also logged as a file artifact according to this format string.
            The same format variables as for ``filename_format`` are available.

            .. seealso:: :meth:`~composer.core.logging.Logger.file_artifact` for file artifact logging.

            Leading slashes (``'/'``) will be stripped.

            To disable logging trace files as file artifacts, set this parameter to ``None``.
        overwrite (bool, optional): Whether to override existing Torch Profiler traces. Defaults to False.

            If False, then the trace folder as determined by ``folder_format`` must be empty.
        use_gzip (bool, optional): Whether to use gzip for the trace. Defaults to False.
            If True, ``'.gz'`` will be appended ``filename_format`` and ``artifact_name_format``
            (if they do not already end in ``'.gz'``).
        record_shapes (bool, optional): Whether to record tensor shapes. Defaults to False.
        profile_memory (bool, optional): Whether to profile memory. Defaults to True.
        with_stack (bool, optional): Whether to record stack info. Defaults to False.
        with_flops (bool, optional): Whether to estimate flops for operators. Defaults to True.
        num_traces_to_keep (int, optional): The number of trace files to keep locally. Defaults to -1.

            If set to -1, then all traces files are kept locally.

            After a trace has been saved and logged as a file artifact, the oldest traces are removed until
            ``num_traces_to_keep`` traces remain. This parameter only controls how many traces are kept locally;
            traces are not deleted from artifact stores.

            It can be useful to set this parameter to ``0`` when using an artifact logger such as the
            :class:`~composer.loggers.object_store_logger.ObjectStoreLogger`. This combination will minimize local
            disk usage by deleting trace files immediately after they have been uploaded to the object store.

    Attributes:
        saved_traces (Dict[Timestamp, List[str]]): A dictionary mapping a save timestamp
            to a list of filepaths corresponding to the traces saved at that time. The list is indexed by
            global rank. These filepaths are valid only on the node for the corresponding rank. This dictionary
            will contain at most ``num_traces_to_keep`` entries.
    """

    def __init__(
        self,
        folder_format: str = '{run_name}/torch_traces',
        filename_format: str = 'ep{epoch}-ba{batch}-rank{rank}.json',
        artifact_name_format: Optional[str] = '{run_name}/torch_traces/ep{epoch}-ba{batch}-rank{rank}.json',
        *,
        overwrite: bool = False,
        use_gzip: bool = False,
        record_shapes: bool = False,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = True,
        num_traces_to_keep: int = -1,
    ) -> None:
        self.overwrite = overwrite
        self.folder_format = folder_format
        if use_gzip and not filename_format.endswith('.gz'):
            filename_format += ".gz"
        self.filename_format = filename_format
        if use_gzip and artifact_name_format is not None and not artifact_name_format.endswith('.gz'):
            artifact_name_format += ".gz"
        self.artifact_name_format = artifact_name_format
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.num_traces_to_keep = num_traces_to_keep
        self.saved_traces = OrderedDict()
        self.profiler: Optional[torch.profiler.profile] = None

    def init(self, state: State, logger: Logger) -> None:
        if state.profiler is None:
            raise RuntimeError(("The Composer Profiler was not enabled, which is required to use the "
                                f"{type(self).__name__}. To enable, set the `prof_schedule` argument of the Trainer."))

        folder_name = format_name_with_dist(self.folder_format, logger.run_name)
        os.makedirs(folder_name, exist_ok=True)
        if not self.overwrite:
            ensure_folder_is_empty(folder_name)

        dist.barrier()

        def scheduler_fn(torch_profiler_step: int) -> TorchProfilerAction:
            del torch_profiler_step  # the torch profiler step is unused. Using the composer timer instead.

            assert state.profiler is not None
            composer_profiler_action = state.profiler.get_action(state)
            if composer_profiler_action == ProfilerAction.ACTIVE_AND_SAVE:
                return TorchProfilerAction.RECORD_AND_SAVE
            if composer_profiler_action == ProfilerAction.ACTIVE:
                return TorchProfilerAction.RECORD
            if composer_profiler_action == ProfilerAction.WARMUP:
                return TorchProfilerAction.WARMUP
            assert composer_profiler_action == ProfilerAction.SKIP, f"unexpected action: {composer_profiler_action}"
            return TorchProfilerAction.NONE

        def handler_fn(prof: torch.profiler.profiler.profile):

            assert state.profiler is not None

            timestamp = state.timer.get_timestamp()

            trace_file_name = os.path.join(
                folder_name,
                format_name_with_dist_and_time(self.filename_format, run_name=logger.run_name, timestamp=timestamp),
            )
            os.makedirs(os.path.dirname(trace_file_name), exist_ok=True)
            prof.export_chrome_trace(trace_file_name)
            state.profiler.record_chrome_json_trace_file(trace_file_name)
            if self.artifact_name_format is not None:
                trace_artifact_name = format_name_with_dist_and_time(self.artifact_name_format,
                                                                     run_name=logger.run_name,
                                                                     timestamp=timestamp)
                trace_artifact_name = trace_artifact_name.lstrip('/')
                logger.file_artifact(LogLevel.BATCH,
                                     artifact_name=trace_artifact_name,
                                     file_path=trace_file_name,
                                     overwrite=self.overwrite)

            if self.num_traces_to_keep >= 0:
                while len(self.saved_traces) > self.num_traces_to_keep:

                    # self.saved_traces is an ordered dict, so the zeroth item will be the oldest checkpoint
                    timestamp, filepaths = next(iter(self.saved_traces.items()))
                    if dist.get_global_rank() < len(filepaths):
                        # Remove this rank's checkpoint
                        os.remove(filepaths[dist.get_global_rank()])
                    del self.saved_traces[timestamp]

        self.profiler = torch.profiler.profile(
            schedule=scheduler_fn,
            on_trace_ready=handler_fn,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
        )
        self.profiler.__enter__()

    def batch_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        assert self.profiler is not None
        self.profiler.add_metadata_json("global_rank", json.dumps(dist.get_global_rank()))
        self.profiler.step()

    def batch_start(self, state: State, logger: Logger) -> None:
        del state  # unused
        assert self.profiler is not None
        logger.data_batch({"profiler/state": self.profiler.current_action.name})

    def close(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
