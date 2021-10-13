# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from composer.core.serializable import Serializable

if TYPE_CHECKING:
    from composer.core import Event, Logger, State


class Algorithm(Serializable, ABC):
    """Base class for algorithms.

    Algorithms are pieces of code which run at specific events in the training
    loop. Algorithms modify the trainer's state, generally with the effect of
    improving the model's quality, or
    increasing the efficiency and throughput of the training loop.

    Algorithms must implement two methods:
    :func:`match`, which returns whether the algorithm should be run given
    the current event and state, and :func:`apply`, which makes an in-place
    change to the State.
    """

    @property
    def find_unused_parameters(self) -> bool:
        """Indicates that the effect of this algorithm may cause some model
        parameters to be unused.

        Used to tell DDP that some parameters will be frozen during
        training and hence it should not expect gradients from them.
        All algorithms which do any kind of parameter freezing should
        override this function to return True.
        """
        return False

    @abstractmethod
    def match(self, event: Event, state: State) -> bool:
        """Determines whether this algorithm should run, given the current
        :class:`Event` and :class:`State`.

        Examples:

        To only run on a specific event:

            >>> return event == Event.BEFORE_LOSS

        Switching based on state attributes:

            >>> return state.epoch > 30 && state.world_size == 1

        See :class:`State` for accessible attributes.

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run now.
        """
        raise NotImplementedError(f'implement match() required for {self.__class__.__name__}')

    @abstractmethod
    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Applies the algorithm to make an in-place change to the State

        Can optionally return an exit code to be stored in a :class:`Trace`.

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
            logger (:class:`Logger`): A logger to use for
                logging algorithm-specific metrics.
        Returns:
            ``int`` or ``None``: exit code that is stored in :class:`Trace`
            and made accessible for debugging.
        """
        raise NotImplementedError(f'implement apply() required for {self.__class__.__name__}')

    def __str__(self) -> str:
        """Returns the class name."""
        return self.__class__.__name__
