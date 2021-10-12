from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from composer.core.serializable import Serializable

if TYPE_CHECKING:
    from composer.core import Event, Logger, State


class Algorithm(Serializable, ABC):
    """
    Base class for algorithms. Algorithms must implement two methods:
    ``match``, which returns whether the algorithm should be run given
    the current event and state, and ``apply``, which makes an in-place
    change to the State.
    """

    @property
    def find_unused_parameters(self) -> bool:
        """
        Used to tell DDP that some parameters will be frozen during
        training and hence it should not expect gradients from them.
        All algorithms which do any kind of parameter freezing should
        override this function to return True.
        """
        return False

    @abstractmethod
    def match(self, event: Event, state: State) -> bool:
        """
        Abstract method to determine whether algorithms should run, given
        the provided ``Event`` and ``State``.

        Examples:
            To only run on a specific event,
            >>> return event == Event.BEFORE_LOSS

            Switching based on event attributes can be done with
            >>> return state.epoch > 30 && state.world_size == 1

            See ``State`` for accessible attributes.

        Args:
            event (Event): the current ``Event``
            state (State): the current ``State``
        Returns:
            bool: True if algorithm should run.
        """
        raise NotImplementedError(f'implement match() required for {self.__class__.__name__}')

    @abstractmethod
    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """
        Applies the algorithm to make in-place changes to the State. Can optionally return
        an exit code that is stored in the ``Trace``.
        Args:
            event (Event): the current ``Event``
            state (State): the current ``State``
        Returns:
            int or None: exit code that is stored in ``Trace`` and accessible for debugging.
        """
        raise NotImplementedError(f'implement apply() required for {self.__class__.__name__}')

    def __str__(self) -> str:
        """ Returns the class name."""
        return self.__class__.__name__
