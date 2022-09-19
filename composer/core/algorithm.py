# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Base class for algorithms that improve a model's quality or efficiency."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from composer.core.serializable import Serializable

if TYPE_CHECKING:
    from composer.core import Event, State
    from composer.loggers import Logger

__all__ = ['Algorithm']


class Algorithm(Serializable, ABC):
    """Base class for algorithms.

    Algorithms are pieces of code which run at specific events (see :class:`.Event`) in the training loop.
    Algorithms modify the trainer's :class:`.State`, generally with the effect of improving the model's quality
    or increasing the efficiency and throughput of the training loop.

    Algorithms must implement the following two methods:
      +----------------+-------------------------------------------------------------------------------+
      | Method         | Description                                                                   |
      +================+===============================================================================+
      | :func:`match`  | returns whether the algorithm should be run given the current                 |
      |                | :class:`.Event` and :class:`.State`.                                          |
      +----------------+-------------------------------------------------------------------------------+
      | :func:`apply`  | Executes the algorithm's code and makes an in-place change                    |
      |                | to the :class:`.State`.                                                       |
      +----------------+-------------------------------------------------------------------------------+
    """

    def __init__(self, *args, **kwargs):  # Stub signature for PyRight
        del args, kwargs  # unused
        pass

    @property
    def find_unused_parameters(self) -> bool:
        """Indicates whether this algorithm may cause some model parameters to be unused. Defaults to False.

        For example, it is used to tell :class:`torch.nn.parallel.DistributedDataParallel` (DDP) that some parameters
        will be frozen during training, and hence it should not expect gradients from them. All algorithms which do any
        kind of parameter freezing should override this function to return ``True``.

        .. note::

           DeepSpeed integration with this function returning True is not tested. It may not work as expected.
        """
        return False

    @property
    def backwards_create_graph(self) -> bool:
        """Return ``True`` to indicate this algorithm requires a second derivative to be computed. Defaults to ``False``.

        If it returns ``True``, ``create_graph=True`` will be passed to :meth:`torch.Tensor.backward` which will result in
        the graph of the gradient also being constructed. This allows the computation of second order derivatives.
        """
        return False

    @staticmethod
    def required_on_load() -> bool:
        """Return `True` to indicate this algorithm is required when loading from a checkpoint which used it."""
        return False

    def state_dict(self) -> Dict[str, Any]:
        return {'repr': self.__repr__()}

    @abstractmethod
    def match(self, event: Event, state: State) -> bool:
        """Determines whether this algorithm should run given the current :class:`.Event` and :class:`.State`.

        Examples:
        To only run on a specific event (e.g., on :attr:`.Event.BEFORE_LOSS`), override match as shown below:

        >>> class MyAlgorithm:
        ...     def match(self, event, state):
        ...         return event == Event.BEFORE_LOSS
        >>> MyAlgorithm().match(Event.BEFORE_LOSS, state)
        True

        To run based on some value of a :class:`.State` attribute, override match as shown below:

        >>> class MyAlgorithm:
        ...     def match(self, event, state):
        ...        return state.timestamp.epoch > 30
        >>> MyAlgorithm().match(Event.BEFORE_LOSS, state)
        False

        See :class:`.State` for accessible attributes.

        Args:
            event (Event): The current event.
            state (State): The current state.

        Returns:
            bool: True if this algorithm should run now.
        """
        raise NotImplementedError(f'implement match() required for {self.__class__.__name__}')

    @abstractmethod
    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Applies the algorithm to make an in-place change to the :class:`.State`.

        Can optionally return an exit code to be stored in a :class:`.Trace`.
        This exit code is made accessible for debugging.

        Args:
            event (Event): The current event.
            state (State): The current state.
            logger (Logger): A logger to use for logging algorithm-specific metrics.

        Returns:
            int or None: exit code that will be stored in :class:`.Trace` and made accessible for debugging.
        """
        raise NotImplementedError(f'implement apply() required for {self.__class__.__name__}')
