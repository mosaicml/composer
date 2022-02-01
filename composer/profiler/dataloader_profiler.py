# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import textwrap
from typing import Iterator, Optional

from composer.core.callback import Callback
from composer.core.profiler import Profiler
from composer.core.state import State
from composer.core.types import Batch, DataLoader, Logger
from composer.datasets.dataloader import WrappedDataLoader


class ProfiledDataLoader(WrappedDataLoader):
    """Wraps a dataloader to record the duration it takes to yield a batch.
    This class should not be instantiated directly.

    Args:
        profiler (Profiler): The profiler instance.
        dataloader (DataLoader): The dataloader to profile.
        name (str): The name for the dataloader.
    """

    def __init__(self, profiler: Profiler, dataloader: DataLoader, name: str) -> None:
        super().__init__(dataloader)
        self._marker = profiler.marker(f"dataloader/{name}", categories=["dataloader"])
        self._iterator: Optional[Iterator[Batch]] = None

    def __iter__(self) -> ProfiledDataLoader:
        self._iterator = iter(self.dataloader)
        return self

    def __next__(self) -> Batch:
        assert self._iterator is not None
        self._marker.start()
        try:
            return next(self._iterator)
        finally:
            self._marker.finish()


class DataloaderProfiler(Callback):

    def init(self, state: State, logger: Logger):
        del logger  # unused
        if state.profiler is None:
            raise RuntimeError(
                textwrap.dedent("""To use the dataloader profiler, state.profiler must be set.
                Make sure to run composer with the profiler -- i.e. with the `--profiler` CLI flag."""))

        if not ProfiledDataLoader.is_dataloader_already_wrapped(state.train_dataloader):
            state.train_dataloader = ProfiledDataLoader(state.profiler, state.train_dataloader, "train")

        if not ProfiledDataLoader.is_dataloader_already_wrapped(state.eval_dataloader):
            state.eval_dataloader = ProfiledDataLoader(state.profiler, state.eval_dataloader, "eval")
