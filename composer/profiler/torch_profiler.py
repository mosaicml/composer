# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Profiler to collect :mod:`torch` performance metrics during training."""

from __future__ import annotations

import json
import os
import textwrap
from typing import TYPE_CHECKING, Optional, OrderedDict

import torch.profiler
from torch.profiler.profiler import ProfilerAction as TorchProfilerAction

from composer.core.callback import Callback
from composer.loggers import Logger
from composer.profiler.profiler_action import ProfilerAction
from composer.utils import (FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, FORMAT_NAME_WITH_DIST_TABLE, dist,
                            ensure_folder_is_empty, format_name_with_dist, format_name_with_dist_and_time)

if TYPE_CHECKING:
    from composer.core import State

__all__ = ['TorchProfiler']


class TorchProfiler(Callback):  # noqa: D101
    __doc__ = f"""Profile the execution using the :class:`PyTorch Profiler <torch.profiler.profile>`.

    Profiling results are stored in TensorBoard format in the directory specified by ``folder``.

    .. note::

        The Composer :class:`.Trainer` automatically creates an instance of this
        :class:`.TorchProfiler` callback whenever any of the PyTorch Profiler arguments
        (``torch_prof_record_shapes``, ``torch_prof_profile_memory``, ``torch_prof_with_stack``, or
        ``torch_prof_with_flops``) are enabled.

        When using the Composer :class:`.Trainer`, one does not need to directly create an
        instance of this :class:`.TorchProfiler` callback.


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
        folder (str, optional): Format string for the folder containing the Torch Profiler trace files.
            Defaults to ``'{{run_name}}/torch_traces'``.

            The following format variables are available:

            {textwrap.indent(FORMAT_NAME_WITH_DIST_TABLE, prefix='            ')}

            For example, if the ``run_name`` is ``'awesome_training_run'``, and the default ``folder`` of
            ``'{{run_name}}/torch_traces'`` is used, Torch Profiler traces will be stored in
            ``'awesome_training_run/torch_traces'``.

        filename (str, optional): A format string describing how to name Torch Profiler trace files.
            Defaults to ``'rank{{rank}}.{{batch}}.pt.trace.json'``.

            At the end of each batch where :meth:`~composer.profiler.Profiler.get_action` returns
            :attr:`~composer.profiler._profiler_action.ProfilerAction.ACTIVE_AND_SAVE`, trace files are saved
            approximately to ``{{folder.format(...)}}/{{filename.format(...)}}``.

            The following format variables are available:

            {textwrap.indent(FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, prefix='            ')}

            Consider the following scenario, where:

            *   The :attr:`~.State.run_name` is ``'awesome-training-run'``.
            *   The default ``trace_folder='{{run_name}}/torch_traces'`` is used.
            *   The default ``name='rank{{rank}}.{{batch}}.pt.trace.json'`` is used.
            *   The current epoch count is ``1``.
            *   The current batch count is ``42``.

            Each rank (process) will save traces to::

                awesome-training-run/torch_traces/ep1-ba42-rank0.json
                awesome-training-run/torch_traces/ep1-ba42-rank1.json
                awesome-training-run/torch_traces/ep1-ba42-rank2.json
                ...

        remote_file_name (str, optional): Format string for a Torch Profiler trace file's remote file name.
            Defaults to ``'{{run_name}}/torch_traces/rank{{rank}}.{{batch}}.pt.trace.json'``.

            Whenever a trace file is saved, it is also uploaded as a file according to this format string.
            The same format variables as for ``filename`` are available.

            .. seealso:: :doc:`Uploading Files</trainer/file_uploading>` for notes for file uploading.

            Leading slashes (``'/'``) will be stripped.

            To disable uploading trace files, set this parameter to ``None``.
        overwrite (bool, optional): Whether to override existing Torch Profiler traces. Defaults to False.

            If False, then the trace folder as determined by ``folder`` must be empty.
        use_gzip (bool, optional): Whether to use gzip for the trace. Defaults to False.
            If True, ``'.gz'`` will be appended ``filename`` and ``remote_file_name``
            (if they do not already end in ``'.gz'``).
        record_shapes (bool, optional): Whether to record tensor shapes. Defaults to False.
        profile_memory (bool, optional): Whether to profile memory. Defaults to True.
        with_stack (bool, optional): Whether to record stack info. Defaults to False.
        with_flops (bool, optional): Whether to estimate flops for operators. Defaults to True.
        num_traces_to_keep (int, optional): The number of trace files to keep locally. Defaults to -1.

            If set to -1, then all traces files are kept locally.

            After a trace has been saved and uploaded, the oldest traces are removed until
            ``num_traces_to_keep`` traces remain. This parameter only controls how many traces are kept locally;
            traces are not deleted from remote file systems.

            It can be useful to set this parameter to ``0`` when using a remote file uploader such as the
            :class:`.RemoteUploaderDownloader`. This combination will minimize local
            disk usage by deleting trace files immediately after they have been uploaded to the object store.

    Attributes:
        saved_traces (List[Tuple[Timestamp, List[pathlib.Path]]]): The trace timestamps and filepaths.

            This list contains tuples of the save timestamp and the trace filepaths.
            This list will have at most ``num_traces_to_keep`` entries. The latest trace
            will be at the end.

            The index of a filepath in each list corresponds to the global rank of the process that wrote that file.
            Each filepath is valid only on the process's (rank's) node.
    """

    def __init__(
        self,
        folder: str = '{run_name}/torch_traces',
        filename: str = 'rank{rank}.{batch}.pt.trace.json',
        remote_file_name: Optional[str] = '{run_name}/torch_traces/rank{rank}.{batch}.pt.trace.json',
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
        self.folder = folder
        if use_gzip and not filename.endswith('.gz'):
            filename += '.gz'
        self.filename = filename
        if use_gzip and remote_file_name is not None and not remote_file_name.endswith('.gz'):
            remote_file_name += '.gz'
        self.remote_file_name = remote_file_name
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.num_traces_to_keep = num_traces_to_keep
        self.saved_traces = OrderedDict()
        self.profiler: Optional[torch.profiler.profile] = None

    def init(self, state: State, logger: Logger) -> None:
        if state.profiler is None:
            raise RuntimeError(('The Composer Profiler was not enabled, which is required to use the '
                                f'{type(self).__name__}. To enable, set the `prof_schedule` argument of the Trainer.'))

        folder_name = format_name_with_dist(self.folder, state.run_name)
        os.makedirs(folder_name, exist_ok=True)
        if not self.overwrite:
            ensure_folder_is_empty(folder_name)

        dist.barrier()

        def scheduler_fn(torch_profiler_step: int) -> TorchProfilerAction:
            del torch_profiler_step  # the torch profiler step is unused. Using the composer timestamp instead.

            assert state.profiler is not None
            composer_profiler_action = state.profiler.schedule(state)
            if composer_profiler_action == ProfilerAction.ACTIVE_AND_SAVE:
                return TorchProfilerAction.RECORD_AND_SAVE
            if composer_profiler_action == ProfilerAction.ACTIVE:
                return TorchProfilerAction.RECORD
            if composer_profiler_action == ProfilerAction.WARMUP:
                return TorchProfilerAction.WARMUP
            assert composer_profiler_action == ProfilerAction.SKIP, f'unexpected action: {composer_profiler_action}'
            return TorchProfilerAction.NONE

        def handler_fn(prof: torch.profiler.profiler.profile):

            assert state.profiler is not None

            timestamp = state.timestamp

            trace_file_name = os.path.join(
                folder_name,
                format_name_with_dist_and_time(self.filename, run_name=state.run_name, timestamp=timestamp),
            )
            trace_file_dirname = os.path.dirname(trace_file_name)
            if trace_file_dirname:
                os.makedirs(trace_file_dirname, exist_ok=True)
            prof.export_chrome_trace(trace_file_name)
            state.profiler.record_chrome_json_trace_file(trace_file_name)
            if self.remote_file_name is not None:
                trace_remote_file_name = format_name_with_dist_and_time(self.remote_file_name,
                                                                        run_name=state.run_name,
                                                                        timestamp=timestamp)
                trace_remote_file_name = trace_remote_file_name.lstrip('/')
                logger.upload_file(remote_file_name=trace_remote_file_name,
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
        self.profiler.add_metadata_json('global_rank', json.dumps(dist.get_global_rank()))
        self.profiler.step()

    def batch_start(self, state: State, logger: Logger) -> None:
        del state  # unused
        assert self.profiler is not None
        logger.log_traces({'profiler/state': self.profiler.current_action.name})

    def close(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
