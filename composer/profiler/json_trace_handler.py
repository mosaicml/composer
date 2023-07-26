# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Outputs profiling data in JSON trace format."""

from __future__ import annotations

import gzip
import json
import os
import pathlib
import queue
import tempfile
import textwrap
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from composer.loggers import Logger
from composer.profiler.json_trace_merger import merge_traces
from composer.profiler.profiler_action import ProfilerAction
from composer.profiler.trace_handler import TraceHandler
from composer.utils import (FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, FORMAT_NAME_WITH_DIST_TABLE, dist,
                            ensure_folder_is_empty, format_name_with_dist, format_name_with_dist_and_time)

if TYPE_CHECKING:
    from composer.core import State, Timestamp

__all__ = ['JSONTraceHandler']


class JSONTraceHandler(TraceHandler):  # noqa: D101
    __doc__ = f"""Records trace events in Chrome JSON trace format.

    See `this document <https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview>`_
    for more information.

    Traces are output to ``output_directory``.  Traces can be visualized using the Chrome Trace Viewer.
    To view in a Google Chrome browser, navigate to ``chrome://tracing`` and load the JSON trace file.

    Args:
        folder (str, optional): Format string for the trace file folder. Defaults to ``'{{run_name}}/traces'``.

            The following format variables are available:

            {textwrap.indent(FORMAT_NAME_WITH_DIST_TABLE, prefix='            ')}

            For example, if the ``run_name`` is ``'awesome_training_run'``, and the default ``folder`` of
            ``'{{run_name}}/traces'`` is used, traces will be stored in ``'awesome_training_run/traces'``.

        filename (str, optional): A format string describing how to name trace files.
            (default: ``'ep{{epoch}}-ba{{batch}}-rank{{rank}}.json'``)

            At the end of each batch where :meth:`~composer.profiler.Profiler.get_action` returns
            :attr:`~composer.profiler._profiler_action.ProfilerAction.ACTIVE_AND_SAVE`, trace files are saved
            approximately to ``{{folder}}/{{filename.format(...)}}``.

            The following format variables are available:

            {textwrap.indent(FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, prefix='            ')}

            Consider the following scenario, where:

            *   The :attr:`~.State.run_name` is ``'awesome-training-run'``
            *   The default ``trace_folder='{{run_name}}/traces'`` is used.
            *   The default ``name='ep{{epoch}}-ba{{batch}}-rank{{rank}}.json'`` is used.
            *   The current epoch count is ``1``.
            *   The current batch count is ``42``.

            Each rank (process) will save traces to::

                awesome-training-run/traces/ep1-ba42-rank0.json
                awesome-training-run/traces/ep1-ba42-rank1.json
                awesome-training-run/traces/ep1-ba42-rank2.json
                ...

        remote_file_name (str, optional): Format string for the trace file's remote name.
            (default: ``'{{run_name}}/traces/ep{{epoch}}-ba{{batch}}-rank{{rank}}.json'``)

            Whenever a trace file is saved, it is also uploaded as a remote file according to this format string.
            The same format variables as for ``filename`` are available.

            .. seealso:: :doc:`Uploading Files</trainer/file_uploading>` for notes for file uploading.

            Leading slashes (``'/'``) will be stripped.

            To disable uploading trace files, set this parameter to ``None``.

        merged_trace_filename (str, optional): Format string for the merged trace filename.
            (default: ``'node{{node_rank}}.json'``)

            Each rank writes a separate trace file at the end of each profiling cycle. However, when visualizing
            traces, it is generally helpful to merge traces together into a single file. This allows the traces
            across all ranks to be shown in a single view. To

            The same format variables as for ``folder`` are available. The merged trace file is saved
            approximately to ``{{folder}}/{{merged_trace_filename.format(...)}}`` on the local rank zero
            process for each node.

            If specified (the default), the local rank zero process merges together all traces files from that node,
            across all profiling cycles, into a single trace file. The merged trace file is written to the filename
            specified by this format string. There will be one merged trace file per node.

            To disable merging, set this parameter to ``None``.

            .. warning::

                Trace merging blocks the training loop. When profiling live training runs, it is recommended to
                disable trace merging by setting this parameter to ``None``. Instead, traces should be merged together
                in a post-processing step. See :mod:`composer.profiler.json_trace_merger` for additional info.

        merged_trace_remote_file_name (str, optional): Format string for the merged trace file's remote file name.
            (default: ``'{{run_name}}/traces/merged_trace.json'``)

            The same format variables as for ``folder`` are available.

            This parameter has no effect if ``merged_trace_filename`` is None.

            To disable uploading merged trace files, set this parameter to ``None``.

        overwrite (bool, optional): Whether to overwrite existing traces. (default: ``False``)
            If ``False``, the :meth:`trace_folder` (as determined by the ``trace_folder`` argument)
            must be empty when training starts.

        num_traces_to_keep (int, optional): The number of traces to keep locally. The oldest traces
            are removed first. Set to ``-1`` to keep all traces locally. (default: ``-1``)

            Traces will be removed after they have been uploaded. For example, when this handler
            is used in conjunction with the :class:`.RemoteUploaderDownloader`, set this
            parameter to ``0`` to immediately delete traces from the local disk after they have been uploaded to
            the object store.

            This parameter only controls how many traces are kept locally; traces are not deleted from
            remote file systems.

    Attributes:
        saved_traces (List[Tuple[Timestamp, List[pathlib.Path]]]): The trace timestamps and filepaths.

            This list contains tuples of the save timestamp and the trace filepaths.
            This list will have at most ``save_num_traces_to_keep`` entries. The latest trace
            will be at the end.

            The index of a filepath in each list corresponds to the global rank of the process that wrote that file.
            Each filepath is valid only on the process's (rank's) node.
    """

    def __init__(
        self,
        folder: str = '{run_name}/traces',
        filename: str = 'ep{epoch}-ba{batch}-rank{rank}.json',
        remote_file_name: Optional[str] = '{run_name}/traces/ep{epoch}-ba{batch}-rank{rank}.json',
        merged_trace_filename: Optional[str] = 'merged_trace.json',
        merged_trace_remote_file_name: Optional[str] = '{run_name}/traces/merged_trace.json',
        *,
        overwrite: bool = False,
        num_traces_to_keep: int = -1,
    ):
        self.folder = folder
        self.overwrite = overwrite
        self.filename = filename
        self.remote_file_name = remote_file_name
        self.merged_trace_filename = merged_trace_filename
        self.merged_trace_remote_file_name = merged_trace_remote_file_name
        self.saved_traces: List[Tuple[Timestamp, List[pathlib.Path]]] = []
        self.num_traces_to_keep = num_traces_to_keep

        self._queue: queue.Queue[str] = queue.Queue()
        self._is_trace_active = False
        self._save_at_batch_end = False

    def init(self, state: State, logger: Logger) -> None:
        del logger  # unused
        trace_folder = format_name_with_dist(self.folder, run_name=state.run_name)

        os.makedirs(trace_folder, exist_ok=True)
        if not self.overwrite:
            ensure_folder_is_empty(trace_folder)
        # Ensure all ranks checked that the folder is empty before proceeding
        # remove any existing merged trace file
        if self.merged_trace_filename is not None:
            merged_trace_filename = os.path.join(
                trace_folder,
                format_name_with_dist(self.merged_trace_filename, state.run_name),
            )
            merged_trace_dirname = os.path.dirname(merged_trace_filename)
            if merged_trace_dirname:
                if os.path.exists(merged_trace_filename):
                    os.remove(merged_trace_filename)
        dist.barrier()

    def batch_start(self, state: State, logger: Logger) -> None:
        del logger  # unusued
        if state.profiler is None:
            raise RuntimeError(('The Composer Profiler was not enabled, which is required to use the '
                                f'{type(self).__name__}. To enable, set the `prof_schedule` argument of the Trainer.'))
        if state.profiler.schedule(state) != ProfilerAction.SKIP and not self._is_trace_active:
            # Starting a new profiling cycle
            wall_clock_ns = time.time_ns()
            self._record_event(
                name='process_name',
                ph='M',  # metadata
                wall_clock_ns=wall_clock_ns,
                tid=os.getpid(),
                pid=dist.get_global_rank(),
                args={'name': f'Rank {dist.get_global_rank()} training loop process'})
            self._record_event(
                name='thread_name',
                ph='M',  # metadata
                wall_clock_ns=wall_clock_ns,
                tid=os.getpid(),
                pid=dist.get_global_rank(),
                args={'name': f'Training Loop'})
            self._record_event(
                name='thread_sort_index',
                ph='M',  # metadata
                wall_clock_ns=wall_clock_ns,
                tid=os.getpid(),
                pid=dist.get_global_rank(),
                args={'sort_index': 0})  # training loop thread should be first
            self._record_event(
                name='global_rank',
                ph='M',  # metadata
                wall_clock_ns=wall_clock_ns,
                tid=os.getpid(),
                pid=dist.get_global_rank(),
                args={'value': dist.get_global_rank()})
            self._record_event(
                name='process_sort_index',
                ph='M',  # metadata
                wall_clock_ns=wall_clock_ns,
                tid=os.getpid(),
                pid=dist.get_global_rank(),
                args={'sort_index': dist.get_global_rank()})  # sort index for processes should be the global rank
            # Synchronize the clocks
            # Each rank will record a timestamp at approxmately the same real world time
            clock_sync_a = time.time_ns()
            dist.barrier()  # syncronize all ranks
            clock_sync_time_ns = time.time_ns()
            dist.barrier()  # another barrier to bound the error
            clock_sync_b = time.time_ns()
            clock_sync_error_bound = clock_sync_b - clock_sync_a
            self._record_event(
                name='clock_sync_timestamp_us',
                ph='M',  # metadata
                wall_clock_ns=wall_clock_ns,
                tid=os.getpid(),
                pid=dist.get_global_rank(),
                args={'value': clock_sync_time_ns // 1000})

            self._record_event(
                name='clock_sync_error_bound',
                ph='M',  # metadata
                wall_clock_ns=wall_clock_ns,
                tid=os.getpid(),
                pid=dist.get_global_rank(),
                args={'value': clock_sync_error_bound // 1000})

            self._is_trace_active = True

        if state.profiler.schedule(state) == ProfilerAction.ACTIVE_AND_SAVE:
            self._save_at_batch_end = True

    def batch_end(self, state: State, logger: Logger) -> None:
        assert state.profiler is not None
        timestamp = state.timestamp
        trace_folder = format_name_with_dist(self.folder, run_name=state.run_name)
        if self._save_at_batch_end:
            # no longer active, but was previously active.
            # Epty the queue and save the trace file
            trace_filename = os.path.join(
                trace_folder,
                format_name_with_dist_and_time(self.filename, state.run_name, timestamp),
            )
            trace_dirname = os.path.dirname(trace_filename)
            if trace_dirname:
                os.makedirs(trace_dirname, exist_ok=True)
            with open(trace_filename, 'w+') as f:
                is_first_line = True
                f.write('[\n')
                while True:
                    try:
                        s = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if not is_first_line:
                        s = ',\n' + s
                    is_first_line = False
                    f.write(s)
                f.write('\n]\n')

            if self.remote_file_name is not None:
                remote_file_name = format_name_with_dist_and_time(self.remote_file_name, state.run_name, timestamp)
                logger.upload_file(remote_file_name=remote_file_name,
                                   file_path=trace_filename,
                                   overwrite=self.overwrite)
            # Gather the filenames
            trace_files = [pathlib.Path(x) for x in dist.all_gather_object(trace_filename)]
            self.saved_traces.append((timestamp, trace_files))

            # Ensure that all traces have been saved.
            dist.barrier()

            if self.merged_trace_filename is not None and dist.get_local_rank() == 0:
                # Merge together all traces from the node into one file
                start_rank = dist.get_global_rank()
                end_rank = dist.get_global_rank() + dist.get_local_world_size()
                trace_files_to_merge = trace_files[start_rank:end_rank]
                merged_trace_filename = os.path.join(
                    trace_folder,
                    format_name_with_dist(
                        self.merged_trace_filename,
                        state.run_name,
                    ),
                )
                merged_trace_dirname = os.path.dirname(merged_trace_filename)
                if merged_trace_dirname:
                    os.makedirs(merged_trace_dirname, exist_ok=True)

                if os.path.exists(merged_trace_filename):
                    # Include the existing merged trace in the new trace
                    with tempfile.NamedTemporaryFile('x+', delete=False) as f:
                        merge_traces(f.name, merged_trace_filename, *trace_files_to_merge)
                        os.rename(f.name, merged_trace_filename)
                else:
                    # Write the trace directly
                    merge_traces(merged_trace_filename, *trace_files_to_merge)

                if self.merged_trace_remote_file_name is not None:
                    merged_trace_remote_file_name = format_name_with_dist(
                        self.merged_trace_remote_file_name,
                        state.run_name,
                    )
                    logger.upload_file(
                        remote_file_name=merged_trace_remote_file_name,
                        file_path=merged_trace_remote_file_name,
                        overwrite=True,
                    )

            # delete old trace files
            if self.num_traces_to_keep >= 0:
                while len(self.saved_traces) > self.num_traces_to_keep:

                    timestamp, checkpoint_filepaths = self.saved_traces[0]
                    if dist.get_global_rank() < len(checkpoint_filepaths):
                        # Remove this rank's trace
                        os.remove(checkpoint_filepaths[dist.get_global_rank()])
                    del self.saved_traces[0]

            self._is_trace_active = False
            self._save_at_batch_end = False

    def process_duration_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        is_start: bool,
        timestamp: Timestamp,
        wall_clock_time_ns: int,
    ) -> None:
        ph = 'B' if is_start else 'E'
        args = {}
        args['epoch'] = timestamp.epoch.value
        args['batch'] = timestamp.batch.value
        self._record_event(
            name=name,
            categories=','.join(categories),
            ph=ph,
            wall_clock_ns=wall_clock_time_ns,
            pid=dist.get_global_rank(),
            args=args,
            tid=os.getpid(),
        )

    def process_instant_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        timestamp: Timestamp,
        wall_clock_time_ns: int,
    ) -> None:
        args = {}
        args['epoch'] = timestamp.epoch.value
        args['batch'] = timestamp.batch.value
        self._record_event(
            name=name,
            categories=','.join(categories),
            ph='i',
            wall_clock_ns=wall_clock_time_ns,
            args=args,
            pid=dist.get_global_rank(),
            tid=os.getpid(),
            s='p',  # mark instant event for at process level
        )

    def process_counter_event(self, name: str, categories: Union[List[str], Tuple[str, ...]], timestamp: Timestamp,
                              wall_clock_time_ns: int, values: Dict[str, Union[int, float]]) -> None:
        self._record_event(
            name=name,
            categories=','.join(categories),
            ph='C',  # counter event
            wall_clock_ns=wall_clock_time_ns,
            pid=dist.get_global_rank(),
            tid=os.getpid(),
            args=values,
        )

    def _record_event(self, name: str, ph: str, wall_clock_ns: int, pid: int, tid: int, categories: str = '', **kwargs):
        """Helper function to record an event in the trace.

        Args:
            name (str): Event name
            categories (str): Comma-seperated string of event categories
            ph (str): Event type. Should be one of the following
                Duration Events: ``B`` (begin), ``E`` (end)
                Complete Events: ``X``
                Instant Events: ``i``
                Counter Events: ``C``
                Async Events: ``b`` (nestable start), ``n`` (nestable instant), ``e`` (nestable end)
                Flow events: ``s`` (start), ``t`` (step), ``f`` (end)
                Sample events: ``P``
                Object Events ``N`` (created), ``O`` (snapshot), ``D`` (destroyed)
                Metadata Events: ``M``
                Memory Dump Events: ``V`` (global), ``v`` (process)
                Mark Events: ``R``
                Clock Sync Events ``c``
            wall_clock_ns (int): Wall clock time, in nanoseconds.
            tid (int): :meth:`threading.get_ident` value for the event
            pid (int): :meth:`os.get_pid` value for the event
            kwargs: Any extra info to record with the event, such as event specific fields.
        """
        data = {
            'name': name,
            'cat': categories,
            'ph': ph,
            'ts': wall_clock_ns // 1000,  # tracing clock timestamp, in microseconds
            'pid': pid,
            'tid': tid,
            **kwargs,
        }
        entry = json.dumps(data, indent=None)
        self._queue.put_nowait(entry)

    def process_chrome_json_trace_file(self, filepath: pathlib.Path) -> None:
        with (gzip.open(filepath, 'rt') if str(filepath).endswith('.gz') else open(filepath, 'r')) as f:
            # It may be an incomplete trace file that is missing the closing ] bracket, as is permitted
            # in the chrome json format spec
            trace_data_str = f.read().strip()
            if trace_data_str.startswith('[') and not trace_data_str.endswith(']'):
                trace_data_str += ']'
            trace_data = json.loads(trace_data_str)

        if isinstance(trace_data, dict):
            event_list = trace_data['traceEvents']
        else:
            event_list = trace_data

        if not isinstance(event_list, list):
            raise TypeError('A trace file should either be a dict or a list')

        for entry in event_list:
            entry['pid'] = dist.get_global_rank()  # override the PID to the global rank
            entry_s = json.dumps(entry, indent=None)
            self._queue.put_nowait(entry_s)
