# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a memory snapshot during an OutOfMemory exception."""
from __future__ import annotations

import dataclasses
import logging
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch.cuda

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import ensure_folder_is_empty, format_name_with_dist, format_name_with_dist_and_time, parse_uri

log = logging.getLogger(__name__)

__all__ = ['OOMObserver']


@dataclass(frozen=True)
class SnapshotFileNameConfig:
    """Configuration for the file names of the memory snapshot visualizations."""
    snapshot_file: str
    trace_plot_file: str
    segment_plot_file: str
    segment_flamegraph_file: str
    memory_flamegraph_file: str

    @classmethod
    def from_file_name(cls, filename: str) -> 'SnapshotFileNameConfig':
        return cls(
            snapshot_file=filename + '_snapshot.pickle',
            trace_plot_file=filename + '_trace_plot.html',
            segment_plot_file=filename + '_segment_plot.html',
            segment_flamegraph_file=filename + '_segment_flamegraph.svg',
            memory_flamegraph_file=filename + '_memory_flamegraph.svg',
        )

    def list_filenames(self) -> list[str]:
        return [getattr(self, field.name) for field in dataclasses.fields(self)]


class OOMObserver(Callback):
    """Generate visualizations of the state of allocated memory during an OutOfMemory exception.

    This callback registers an observer with the allocator that will be called everytime it is about to raise an OutOfMemoryError before any memory has been release while unwinding the exception. OOMObserver is attached to the Trainer at init stage. The visualizations include a snapshot of the memory state, a trace plot, a segment plot, a segment flamegraph, and a memory flamegraph.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import OOMObserver
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration="1ep",
            ...     callbacks=[OOMObserver()],
            ... )

    .. note::
        OOMObserver is only supported for GPU devices.

    Args:
        max_entries (int, optional): Maximum number of memory alloc/free events to record. Defaults to 100000.
        folder (str, optional): A format string describing the folder containing the memory visualization files.
            Defaults to ``'{{run_name}}/torch_traces'``.
        filename (str, optional): A format string describing the prefix used to name the memory visualization files.
            Defaults to ``'rank{{rank}}_oom'``.
        remote_file_name (str, optional): A format string describing the prefix for the memory visualization remote file name.
            Defaults to ``'{{run_name}}/oom_traces/rank{{rank}}_oom'``.

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
        max_entries: int = 100000,
        folder: str = '{run_name}/torch_traces',
        filename: str = 'rank{rank}_oom',
        remote_file_name: Optional[str] = '{run_name}/oom_traces/rank{rank}_oom',
        overwrite: bool = False,
    ) -> None:
        self.max_entries = max_entries
        self.folder = folder
        self.folder_name = None
        self.filename = filename
        self.remote_file_name = remote_file_name
        self.overwrite = overwrite
        if remote_file_name:
            self.remote_file_name = remote_file_name
            _, _, self.remote_path_in_bucket = parse_uri(remote_file_name)
        else:
            self.remote_path_in_bucket = None

        self._enabled = True
        self.filename_config: Optional[SnapshotFileNameConfig] = None

    def init(self, state: State, logger: Logger) -> None:
        if not self._enabled:
            return
        # Not relying on `torch.cuda.is_available()` since the model could be on CPU.
        model_device = next(state.model.parameters()).device

        if model_device.type not in ('cuda', 'meta'):
            warnings.warn(
                f'OOMObserver only works on CUDA devices, but the model is on {model_device.type}. Disabling OOMObserver.',
            )
            self._enabled = False
        else:
            self.folder_name = format_name_with_dist(self.folder, state.run_name)
            os.makedirs(self.folder_name, exist_ok=True)
            if not self.overwrite:
                ensure_folder_is_empty(self.folder_name)

        def oom_observer(device: int, alloc: int, device_alloc: int, device_free: int):
            # Snapshot right after an OOM happened
            log.warning('Out Of Memory (OOM) observed')

            assert self.filename
            assert self.folder_name, 'folder_name must be set in init'
            filename = Path(self.folder_name) / Path(
                format_name_with_dist_and_time(self.filename, run_name=state.run_name, timestamp=state.timestamp),
            )

            try:
                self.filename_config = SnapshotFileNameConfig.from_file_name(str(filename))
                log.info(f'Dumping OOMObserver visualizations')

                snapshot = torch.cuda.memory._snapshot()
                # No data was recorded - avoids a `ValueError` in `trace_plot`
                if all(len(t) == 0 for t in snapshot['device_traces']):
                    log.info(f'No allocation is recorded in memory snapshot)')
                    return

                with open(self.filename_config.snapshot_file, 'wb') as fd:
                    pickle.dump(snapshot, fd)

                with open(self.filename_config.trace_plot_file, 'w+') as fd:
                    fd.write(torch.cuda._memory_viz.trace_plot(snapshot))  # type: ignore

                with open(self.filename_config.segment_plot_file, 'w+') as fd:
                    fd.write(torch.cuda._memory_viz.segment_plot(snapshot))  # type: ignore

                with open(self.filename_config.segment_flamegraph_file, 'w+') as fd:
                    fd.write(torch.cuda._memory_viz.segments(snapshot))  # type: ignore

                with open(self.filename_config.memory_flamegraph_file, 'w+') as fd:
                    fd.write(torch.cuda._memory_viz.memory(snapshot))  # type: ignore

                log.info(f'Saved memory visualizations to local files with prefix = {filename} during OOM')

                if self.remote_path_in_bucket is not None:
                    for f in self.filename_config.list_filenames():
                        base_file_name = os.path.basename(f)
                        remote_file_name = os.path.join(self.remote_path_in_bucket, base_file_name)
                        remote_file_name = remote_file_name.lstrip('/')  # remove leading slashes
                        log.info(f'Uploading memory visualization to remote: {remote_file_name} from {f}')
                        try:
                            logger.upload_file(remote_file_name=remote_file_name, file_path=f, overwrite=self.overwrite)
                        except FileExistsError as e:
                            raise FileExistsError(
                                f'Uploading memory visualizations failed with error: {e}. overwrite was set to {self.overwrite}. To overwrite memory visualizations with Trainer, set save_overwrite to True.',
                            ) from e

            except Exception as e:
                log.error(f'Failed to capture memory snapshot {e}')

        if self._enabled:
            torch.cuda.memory._record_memory_history(
                True,  # type: ignore
                trace_alloc_max_entries=self.max_entries,
                trace_alloc_record_context=True,
            )
            torch._C._cuda_attach_out_of_memory_observer(oom_observer)  # type: ignore
            log.info('OOMObserver is enabled and registered')
