# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""

"""
import warnings
from typing import Any, Sequence, TypeVar

from composer.algorithms.warnings import NotIntendedUseWarning
from composer.core import Algorithm, Event

T = TypeVar('T')


def sort_to_front(list_to_sort: Sequence[T], cls: Any) -> Sequence[T]:
    """Helper function to sort instances of a provided class to the front.

    Example:

        .. doctest::

            >>> sort_to_front([1, 'b', 3], str)
            ['b', 1, 3]

    Args:
        list_to_sort: list of objects to sort
        cls: sorts all objects of this class to the front

    Returns:
        sorted_list: Sorted List

    """
    return sorted(list_to_sort, key=lambda x: not isinstance(x, cls))


def sort_to_back(list_to_sort: Sequence[T], cls: Any) -> Sequence[T]:
    """Helper function to sort instances of a provided class to the back.

    Example:

        .. doctest::

            >>> sort_to_back([1, 'b', 3], str)
            [1, 3, 'b']

    Args:
        list_to_sort: list of objects to sort
        cls: sorts all objects of this class to the back

    Returns:
        sorted_list: Sorted List

    """
    return sorted(list_to_sort, key=lambda x: isinstance(x, cls))


def sort_selective_backprop_first(algorithms: Sequence[Algorithm], event: Event):
    """Selective Backprop should run before any algorithms modify the loss."""
    from composer.algorithms import SelectiveBackprop
    return sort_to_front(algorithms, cls=SelectiveBackprop)


def sort_fused_layernorm_last(algorithms: Sequence[Algorithm], event: Event):
    """FusedLayerNorm should run after other algorithms that add LayerNorms (e.g. GatedLinearUnits)."""
    from composer.algorithms import FusedLayerNorm
    return sort_to_back(algorithms, cls=FusedLayerNorm)


def set_filo_order(algorithms: Sequence[Algorithm], event: Event):
    """Establish a FILO order of algorithms ``before_`` and ``after_`` events.

        For example, algorithms will run in order ABCD during ``before_loss``,
        and in DCBA during ``after_loss``. Algorithms can then 'undo' their effects
        upon the exit of an event.

        Events with the pattern ``_start`` or ``_end`` will not be affected.
    """
    if event.name.startswith('after_'):
        return list(reversed(algorithms))

    return algorithms


def warn_if_multiple_loss_interpolation(algorithms: Sequence[Algorithm], event: Event):
    """Multiple algorithms that interpolate the loss may have unexpected behavior."""
    is_interpolate = [a for a in algorithms if hasattr(a, 'interpolate_loss') and a.interpolate_loss]
    if len(is_interpolate) > 1:
        warnings.warn(
            NotIntendedUseWarning(
                f'Multiple algorithms interpolating the loss can lead to unexpected behavior: {is_interpolate}'))
