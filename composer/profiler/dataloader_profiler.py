# Copyright 2021 MosaicML. All Rights Reserved.

"""Profiler to measure the time it takes the data loader to return a batch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Optional

from composer.core.callback import Callback
from composer.datasets.dataloader import WrappedDataLoader

if TYPE_CHECKING:
    from composer.core.state import State
    from composer.core.types import Batch, DataLoader
    from composer.loggers import Logger
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
    """Profile a DataLoader.

    This callback measures the latency it takes for the DataLoader to yield a batch.

    .. note::

        The Composer :class:`~composer.trainer.trainer.Trainer` automatically creates an instance of this
        :class:`.DataLoaderProfiler` callback whenever the profiler is enabled.

        When using the Composer :class:`~composer.trainer.trainer.Trainer`, one does not need to directly create an
        instance of this :class:`.DataLoaderProfiler` callback.
    """

    def fit_start(self, state: State, logger: Logger):
        del logger  # unused
        if state.profiler is None:
            raise RuntimeError(("The Composer Profiler was not enabled, which is required to use the "
                                f"{type(self).__name__}. To enable, set the `prof_schedule` argument of the Trainer."))

        if not _ProfiledDataLoader.is_dataloader_already_wrapped(state.train_dataloader):
            state.train_dataloader = _ProfiledDataLoader(state.profiler, state.train_dataloader, "train")

        for evaluator in state.evaluators:

            if not _ProfiledDataLoader.is_dataloader_already_wrapped(evaluator.dataloader.dataloader):
                evaluator.dataloader.dataloader = _ProfiledDataLoader(state.profiler, evaluator.dataloader.dataloader,
                                                                      evaluator.label)
