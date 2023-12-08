# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Profiler to collect :mod:`torch` performance metrics during training."""

from __future__ import annotations

import json
import logging
import os
import textwrap
from typing import TYPE_CHECKING, Optional, OrderedDict

import torch.cuda
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

log = logging.getLogger(__name__)


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

        pip install tensorboard torch_tb_profiler
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

                awesome-training-run/torch_traces/ep1-ba42-rank0.pt.trace.json
                awesome-training-run/torch_traces/ep1-ba42-rank1.pt.trace.json
                awesome-training-run/torch_traces/ep1-ba42-rank2.pt.trace.json
                ...

        remote_file_name (str, optional): Format string for a Torch Profiler trace file's remote file name.
            Defaults to ``'{{run_name}}/torch_traces/rank{{rank}}.{{batch}}.pt.trace.json'``.

            Whenever a trace file is saved, it is also uploaded as a file according to this format string.
            The same format variables as for ``filename`` are available.

            .. seealso:: :doc:`Uploading Files</trainer/file_uploading>` for notes for file uploading.

            Leading slashes (``'/'``) will be stripped.

            To disable uploading trace files, set this parameter to ``None``.
        memory_filename (str, optional): A format string describing how to name Torch Profiler memory trace files.
            Defaults to ``'rank{{rank}}.{{batch}}.pt.trace.memory.html'``.

            At the end of each batch where :meth:`~composer.profiler.Profiler.get_action` returns
            :attr:`~composer.profiler._profiler_action.ProfilerAction.ACTIVE_AND_SAVE`, trace files are saved
            approximately to ``{{folder.format(...)}}/{{memory_filename.format(...)}}``.

            The following format variables are available:

            {textwrap.indent(FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, prefix='            ')}

            Consider the following scenario, where:

            *   The :attr:`~.State.run_name` is ``'awesome-training-run'``.
            *   The default ``trace_folder='{{run_name}}/torch_traces'`` is used.
            *   The default ``name='rank{{rank}}.{{batch}}.pt.trace.memory.html'`` is used.
            *   The current epoch count is ``1``.
            *   The current batch count is ``42``.

            Each rank (process) will save traces to::

                awesome-training-run/torch_traces/ep1-ba42-rank0.pt.trace.memory.html
                awesome-training-run/torch_traces/ep1-ba42-rank1.pt.trace.memory.html
                awesome-training-run/torch_traces/ep1-ba42-rank2.pt.trace.memory.html
                ...

        memory_remote_file_name (str, optional): Format string for a Torch Profiler memory trace file's remote file name.
            Defaults to ``'{{run_name}}/torch_traces/rank{{rank}}.{{batch}}.pt.trace.memory.json'``.

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
        memory_filename: Optional[str] = 'rank{rank}.{batch}.pt.trace.memory.html',
        memory_remote_file_name: Optional[
            str] = '{run_name}/torch_memory_traces/rank{rank}.{batch}.pt.trace.memory.html',
        overwrite: bool = False,
        use_gzip: bool = False,
        record_shapes: bool = False,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = True,
        num_traces_to_keep: int = -1,
        memory_custom_plot: bool = False,
    ) -> None:
        self.overwrite = overwrite
        self.folder = folder
        if use_gzip:
            if not filename.endswith('.gz'):
                filename += '.gz'
            if memory_filename is not None and not memory_filename.endswith('.html'):
                memory_filename += '.gz'
        self.filename = filename
        self.memory_filename = memory_filename
        if use_gzip:
            if remote_file_name is not None and not remote_file_name.endswith('.gz'):
                remote_file_name += '.gz'
            if memory_remote_file_name is not None and not memory_remote_file_name.endswith('.gz'):
                memory_remote_file_name += '.gz'
        self.remote_file_name = remote_file_name
        self.memory_remote_file_name = memory_remote_file_name
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.num_traces_to_keep = num_traces_to_keep
        self.saved_traces = OrderedDict()
        self.profiler: Optional[torch.profiler.profile] = None
        self.memory_custom_plot = memory_custom_plot

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
            if not (self.filename is None and self.memory_filename is not None):
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

            log.debug(f'Memory profiler enabled: {self.memory_filename if self.memory_filename else False}')
            if self.memory_filename is not None:
                memory_trace_file_name = os.path.join(
                    folder_name,
                    format_name_with_dist_and_time(self.memory_filename, run_name=state.run_name, timestamp=timestamp),
                )
                log.debug(f'Saving memory trace to {memory_trace_file_name}')
                memory_trace_file_dirname = os.path.dirname(memory_trace_file_name)
                if memory_trace_file_dirname:
                    os.makedirs(memory_trace_file_dirname, exist_ok=True)
                if self.memory_custom_plot:
                    from base64 import b64encode
                    from os import remove
                    from tempfile import NamedTemporaryFile

                    import matplotlib.pyplot as plt
                    import numpy as np
                    from torch.profiler._memory_profiler import (_CATEGORY_TO_COLORS, _CATEGORY_TO_INDEX,
                                                                 MemoryProfileTimeline)

                    # Construct the memory timeline plot data
                    mem_tl = MemoryProfileTimeline(prof._memory_profile())

                    def export_memory_timeline_html(mem_tl, path, device, figsize=(20, 12), title=None) -> None:
                        # Check if user has matplotlib installed, return gracefully if not.
                        import importlib.util

                        matplotlib_spec = importlib.util.find_spec('matplotlib')
                        if matplotlib_spec is None:
                            print('export_memory_timeline_html failed because matplotlib was not found.')
                            return

                        mt = mem_tl._coalesce_timeline(device)
                        times, sizes = np.array(mt[0]), np.array(mt[1])
                        stacked = np.cumsum(sizes, axis=1) / 1024**3
                        max_memory_allocated = torch.cuda.max_memory_allocated()
                        max_memory_reserved = torch.cuda.max_memory_reserved()

                        # Plot memory timeline as stacked data
                        fig = plt.figure(figsize=figsize, dpi=80)
                        axes = fig.gca()
                        for category, color in _CATEGORY_TO_COLORS.items():
                            i = _CATEGORY_TO_INDEX[category]
                            axes.fill_between(times / 1e3, stacked[:, i], stacked[:, i + 1], color=color, alpha=0.7)
                        fig.legend(['Unknown' if i is None else i.name for i in _CATEGORY_TO_COLORS])
                        axes.set_xlabel('Time (us)')
                        axes.set_ylabel('Memory (GB)')
                        _, end = axes.get_ylim()
                        axes.grid(True)
                        axes.set_yticks(np.arange(0, end, 1))
                        title = '\n\n'.join(([title] if title else []) + [
                            f'Max memory allocated: {max_memory_allocated/(10**9):.2f} GB \n'
                            f'Max memory reserved: {max_memory_reserved/(10**9):.2f} GB'
                        ])
                        axes.set_title(title)

                        # Embed the memory timeline image into the HTML file
                        tmpfile = NamedTemporaryFile('wb', suffix='.png', delete=False)
                        tmpfile.close()
                        fig.savefig(tmpfile.name, format='png')

                        with open(tmpfile.name, 'rb') as tmp:
                            encoded = b64encode(tmp.read()).decode('utf-8')
                            html = f"""<html>
                                    <head><meta charset="utf-8" /><title>GPU Memory Timeline HTML</title></head>
                                    <body>
                                    <img src='data:image/png;base64,{encoded}'>
                                    </body>
                                    </html>"""

                            with open(path, 'w') as f:
                                f.write(html)
                        remove(tmpfile.name)

                    export_memory_timeline_html(mem_tl, memory_trace_file_name, torch.cuda.current_device())
                else:
                    prof.export_memory_timeline(memory_trace_file_name, str(torch.cuda.current_device()))
                log.debug(f'Uploaded memory trace to {self.memory_remote_file_name}')
                if self.memory_remote_file_name is not None:
                    memory_trace_remote_file_name = format_name_with_dist_and_time(self.memory_remote_file_name,
                                                                                   run_name=state.run_name,
                                                                                   timestamp=timestamp)
                    memory_trace_remote_file_name = memory_trace_remote_file_name.lstrip('/')
                    log.debug(
                        f'Uploading memory trace to {memory_trace_remote_file_name} from {memory_trace_file_name}')
                    logger.upload_file(remote_file_name=memory_trace_remote_file_name,
                                       file_path=memory_trace_file_name,
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
            log.info(self.profiler.key_averages().table(sort_by='cpu_time_total', row_limit=20))
            if self.profile_memory:
                log.info(self.profiler.key_averages().table(sort_by='self_cpu_memory_usage', row_limit=20))
            if torch.profiler.ProfilerActivity.CUDA in self.profiler.activities:
                log.info(self.profiler.key_averages().table(sort_by='cuda_time_total', row_limit=20))
                if self.profile_memory:
                    log.info(self.profiler.key_averages().table(sort_by='self_cuda_memory_usage', row_limit=20))
            self.profiler.__exit__(None, None, None)
            self.profiler = None
