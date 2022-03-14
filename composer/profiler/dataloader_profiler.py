# Copyright 2021 MosaicML. All Rights Reserved.

"""Profiler to measure the time it takes the data loader to return a batch."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Iterator, Optional

from composer.core.callback import Callback
from composer.datasets.dataloader import WrappedDataLoader

if TYPE_CHECKING:
    from composer.core.state import State
    from composer.core.types import Batch, DataLoader, Logger
    from composer.profiler import Profiler

__all__ = ["DataLoaderProfiler"]


class _ProfiledDataLoader(WrappedDataLoader):
    """Wraps a dataloader to record the duration it takes to yield a batch. This class should not be instantiated
    directly.

    Args:
        profiler (Profiler): The profiler instance.
        dataloader (DataLoader): The dataloader to profile.
        name (str): The name for the dataloader.
    """

    def __init__(self, profiler: Profiler, dataloader: DataLoader, name: str) -> None:
        super().__init__(dataloader)
        self._marker = profiler.marker(f"dataloader/{name}", categories=["dataloader"])
        self._iterator: Optional[Iterator[Batch]] = None

    def __iter__(self) -> _ProfiledDataLoader:
        self._iterator = iter(self.dataloader)
        return self

    def __next__(self) -> Batch:
        assert self._iterator is not None
        self._marker.start()
        try:
            return next(self._iterator)
        finally:
            self._marker.finish()


class DataLoaderProfiler(Callback):
    """Records the time it takes the data loader to return a batch.

    When used with the Composer :class:`.Trainer`\\, the data loader profiler is enabled if profiling is enabled.

    Works by wrapping the original training and evaluation data loaders and uses the :class:`.Marker` API to record
    the latency of the wrapped data loader.

    The profiler is implemented as a :class:`.Callback` and accesses the training and evaluation data loaders through
    :class:`.State`\\.

    .. note::

        The Composer :class:`.Trainer` creates an instance of :class:`.TorchProfiler` when ``tensorboard_trace_handler_dir`` is provided.
        The user should not create and directly register an instance of :class:`.TorchProfiler` when using the Composer :class:`.Trainer`\\.
    """

    def fit_start(self, state: State, logger: Logger):
        del logger  # unused
        if state.profiler is None:
            raise RuntimeError(
                textwrap.dedent("""To use the dataloader profiler, state.profiler must be set.
                Make sure to run composer with the profiler -- i.e. with the `--profiler` CLI flag."""))

        if not _ProfiledDataLoader.is_dataloader_already_wrapped(state.train_dataloader):
            state.train_dataloader = _ProfiledDataLoader(state.profiler, state.train_dataloader, "train")

        for evaluator in state.evaluators:

            if not _ProfiledDataLoader.is_dataloader_already_wrapped(evaluator.dataloader.dataloader):
                evaluator.dataloader.dataloader = _ProfiledDataLoader(state.profiler, evaluator.dataloader.dataloader,
                                                                      evaluator.label)
