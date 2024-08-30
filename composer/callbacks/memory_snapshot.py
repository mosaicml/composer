# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log memory snapshot during training."""
import logging
import os
import pickle
import warnings
from typing import Optional, Union

import torch.cuda

from composer import State
from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from composer.utils import ensure_folder_is_empty, format_name_with_dist, format_name_with_dist_and_time, parse_uri

log = logging.getLogger(__name__)

__all__ = ['MemorySnapshot']


class MemorySnapshot(Callback):
    """Logs the memory snapshot of the model.

    This callback calls the torch memory snapshot API (see :func:`torch.cuda.memory._snapshot`) to record a model's tensor memory allocation over a user defined interval (only once through time [skip_batches, skip_batches + interval]). This provides a fine-grained GPU memory visualization for debugging GPU OOMs. Captured memory snapshots will show memory events including allocations, frees and OOMs, along with their stack traces over one interval.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import MemorySnapshot
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration="1ep",
            ...     callbacks=[MemorySnapshot()],
            ... )

    .. note::
        Memory snapshot is only supported for GPU devices.

    Args:
        skip_batches (int, optional): Number of batches to skip before starting recording memory snapshot. Defaults to 1.
        interval (Union[int, str, Time], optional): Time string specifying how long to record the tensor allocation.
            For example, ``interval='3ba'`` means 3 batches are recorded. Default: '3ba'.
        max_entries (int, optional): Maximum number of memory alloc/free events to record. Defaults to 100000.
        folder (str, optional): A format string describing the folder containing the memory snapshot files.
            Defaults to ``'{{run_name}}/torch_traces'``.
        filename (str, optional): A format string describing the prefix used to name the memory snapshot files.
            Defaults to ``'rank{{rank}}.{{batch}}.memory_snapshot'``.
        remote_file_name (str, optional): A format string describing the prefix for the memory snapshot remote file name.
            Defaults to ``'{{run_name}}/torch_traces/rank{{rank}}.{{batch}}.memory_snapshot'``.

            Whenever a trace file is saved, it is also uploaded as a file according to this format string.
            The same format variables as for ``filename`` are available.

            .. seealso:: :doc:`Uploading Files</trainer/file_uploading>` for notes for file uploading.

            Leading slashes (``'/'``) will be stripped.

            To disable uploading trace files, set this parameter to ``None``.
        overwrite (bool, optional): Whether to override existing memory snapshots. Defaults to False.

            If False, then the trace folder as determined by ``folder`` must be empty.
    """

    def __init__(
        self,
        skip_batches: int = 1,
        interval: Union[int, str, Time] = '3ba',
        max_entries: int = 100000,
        folder: str = '{run_name}/torch_traces',
        filename: str = 'rank{rank}.{batch}.memory_snapshot',
        remote_file_name: Optional[str] = '{run_name}/torch_memory_traces',
        overwrite: bool = False,
    ) -> None:
        self.batches_left_to_skip = skip_batches
        # Check that the interval timestring is parsable and convert into time object
        self.interval = Time.from_input(interval, TimeUnit.BATCH)
        self.max_entries = max_entries
        self.folder = folder
        self.folder_name = None
        self.filename = filename
        self.remote_file_name = remote_file_name
        self.overwrite = overwrite
        self._start_time = None
        if remote_file_name:
            self.remote_file_name = remote_file_name
            _, _, self.remote_path_in_bucket = parse_uri(remote_file_name)
        else:
            self.remote_path_in_bucket = None
        self._enabled = True

    def init(self, state: State, logger: Logger) -> None:
        if not self._enabled:
            return
        # Not relying on `torch.cuda.is_available()` since the model could be on CPU.
        model_device = next(state.model.parameters()).device

        if model_device.type not in ('cuda', 'meta'):
            warnings.warn(f'The memory snapshot only works on CUDA devices, but the model is on {model_device.type}.')
            self._enabled = False
        else:
            self.folder_name = format_name_with_dist(self.folder, state.run_name)
            os.makedirs(self.folder_name, exist_ok=True)
            if not self.overwrite:
                ensure_folder_is_empty(self.folder_name)

    def batch_start(self, state: State, logger: Logger) -> None:
        if self._enabled and self._start_time is None and self.batches_left_to_skip == 0:
            self.start_record_memory_history()
            self._start_time = state.timestamp.get(self.interval.unit).value

    def batch_end(self, state: State, logger: Logger) -> None:
        if not self._enabled:
            return

        if self.batches_left_to_skip > 0:
            self.batches_left_to_skip -= 1
            return
        assert self._start_time is not None

        if state.timestamp.get(self.interval.unit).value == (self._start_time + self.interval.value):
            self.export_memory_snapshot(state, logger)
            self.stop_record_memory_history()
            self._start_time = None
            self._enabled = False

    def start_record_memory_history(self) -> None:

        log.info('Starting snapshot record_memory_history')
        torch.cuda.memory._record_memory_history(
            'all',  # type: ignore
            max_entries=self.max_entries,
        )

    def stop_record_memory_history(self) -> None:

        log.info('Stopping snapshot record_memory_history')
        torch.cuda.memory._record_memory_history(False)  # type: ignore

    def export_memory_snapshot(self, state: State, logger: Logger) -> None:
        assert self.filename
        assert self.folder_name, 'folder_name must be set in init'
        filename = os.path.join(
            self.folder_name,
            format_name_with_dist_and_time(self.filename, run_name=state.run_name, timestamp=state.timestamp),
        )
        try:
            snapshot_file = filename + '.pickle'
            trace_plot_file = filename + '.html'
            log.info(f'Saving memory snapshot files')

            snapshot = torch.cuda.memory._snapshot()
            # No data was recorded - avoids a `ValueError` in `trace_plot`
            if all(len(t) == 0 for t in snapshot['device_traces']):
                log.info(f'No allocation is recorded in memory snapshot)')
                return

            with open(snapshot_file, 'wb') as fd:
                pickle.dump(snapshot, fd)

            with open(trace_plot_file, 'w+') as fd:
                fd.write(torch.cuda._memory_viz.trace_plot(snapshot))  # type: ignore

            log.info(f'Saved memory snapshot to local files with prefix = {filename}')

            if self.remote_path_in_bucket is not None:
                for f in [snapshot_file, trace_plot_file]:
                    remote_file_name = os.path.join(self.remote_path_in_bucket, os.path.basename(f)).lstrip('/')
                    log.info(f'Uploading memory snapshot to remote: {remote_file_name} from {f}')
                    try:
                        logger.upload_file(remote_file_name=remote_file_name, file_path=f, overwrite=self.overwrite)
                    except FileExistsError as e:
                        raise FileExistsError(
                            f'Uploading memory snapshot failed with error: {e}. overwrite was set to {self.overwrite}. To overwrite memory snapshot with Trainer, set `overwrite` to True.',
                        ) from e
        except Exception as e:
            log.error(f'Failed to capture memory snapshot {e}')
