# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Algorithm Passes reorder or modify the execution of algorithms by the Engine.

The order in which algorithms are run matters significantly during composition. For example, the
:class:`.SelectiveBackprop` algorithm runs on the :attr:`.Event.AFTER_DATALOADER` event and must run before
any data augmentations. :class:`.Engine` runs re-ordering passes to resolve such ordering issues or conflicts.

These modifications are represented as algorithm passes, which are functions that modify a list of algorithms.

For example, an algorithm pass that ensures a certain algorithm runs last, would be implemented as:

.. code-block:: python

   def run_last(algorithms: Sequence[Algorithm], event: Event) -> Sequence[Algorithm]:
      algorithms = sorted(algorithms, key=lambda x: isinstance(x, MyAlgorithm))

The passes in this module are registered by default into :class:`.Engine`.
"""
import warnings
from typing import Any, Callable, Sequence, TypeVar

from composer.core.algorithm import Algorithm
from composer.core.event import Event

T = TypeVar('T')

AlgorithmPass = Callable[[Sequence[Algorithm], Event], Sequence[Algorithm]]


def sort_to_front(list_to_sort: Sequence[T], cls: Any) -> Sequence[T]:
    """Helper function to sort instances of a provided class to the front.

    Example:

        .. testsetup::

            from composer.core.passes import sort_to_front

        .. doctest::

            >>> sort_to_front([1, 'b', 3], str)
            ['b', 1, 3]

    Args:
        list_to_sort: list of objects to sort
        cls: sorts all objects of this class to the front

    Returns:
        sorted_list: Sorted list

    """
    return sorted(list_to_sort, key=lambda x: not isinstance(x, cls))


def sort_to_back(list_to_sort: Sequence[T], cls: Any) -> Sequence[T]:
    """Helper function to sort instances of a provided class to the back.

    Example:

        .. testsetup::

            from composer.core.passes import sort_to_back

        .. doctest::

            >>> sort_to_back([1, 'b', 3], str)
            [1, 3, 'b']

    Args:
        list_to_sort: list of objects to sort
        cls: sorts all objects of this class to the back

    Returns:
        sorted_list: Sorted list

    """
    return sorted(list_to_sort, key=lambda x: isinstance(x, cls))


def sort_selective_backprop_first(algorithms: Sequence[Algorithm], event: Event) -> Sequence[Algorithm]:
    """Selective Backprop should run before any algorithms modify the loss.

    :class:`.SelectiveBackprop` runs after the dataloader returns the batch and executes an extra forward pass to rank
    and prune the examples in the batch by loss. To ensure a clean estimate of loss, :class:`.SelectiveBackprop` should
    run before any other data augmentations (e.g., :class:`.MixUp`) on the :attr:`.Event.AFTER_DATALOADER` event.

    """
    from composer.algorithms import SelectiveBackprop
    return sort_to_front(algorithms, cls=SelectiveBackprop)


def sort_low_precision_layernorm_last(
    algorithms: Sequence[Algorithm],
    event: Event,
) -> Sequence[Algorithm]:  #noqa: D403
    """LowPrecisionLayerNorm should run after other algorithms that add LayerNorms (e.g. GatedLinearUnits).

    This ensures that all LayerNorms are converted to the intended precision.

    """
    from composer.algorithms import LowPrecisionLayerNorm
    return sort_to_back(algorithms, cls=LowPrecisionLayerNorm)


def set_filo_order(algorithms: Sequence[Algorithm], event: Event) -> Sequence[Algorithm]:
    """Establish a FILO order of algorithms ``before_`` and ``after_`` events.

    For the events that follow the ``before_*`` and ``after_*`` pattern (e.g., :attr:`.Event.BEFORE_LOSS`
    and :attr:`.Event.AFTER_LOSS), the ordering of algorithms is reversed for the ``after_*`` events.
    For example, four given algorithms ``A``, ``B``, ``C``, and ``D`` will run in ``ABCD`` ordering on
    the ``before_*`` event while ``DCBA`` ordering on the ``after_*`` event.

    This allows algorithms to "clean up" their changes. For example, :class:`.LabelSmoothing` will smooth the labels
    upon the :attr:`.Event.BEFORE_LOSS` event and then restore the original unsmoothed labels on the
    :attr:`.Event.AFTER_LOSS` event.

    Events with the pattern ``_start`` or ``_end`` will not be affected.
    """
    if event.name.startswith('AFTER_'):
        return list(reversed(algorithms))

    return algorithms


def warn_if_multiple_loss_interpolation(algorithms: Sequence[Algorithm], event: Event) -> Sequence[Algorithm]:
    """Multiple algorithms that interpolate the loss may have unexpected behavior."""
    from composer.algorithms.warnings import NotIntendedUseWarning

    is_interpolate = [a for a in algorithms if hasattr(a, 'interpolate_loss') and a.interpolate_loss]  # type: ignore
    if len(is_interpolate) > 1:
        warnings.warn(
            NotIntendedUseWarning(
                f'Multiple algorithms interpolating the loss can lead to unexpected behavior: {is_interpolate}',
            ),
        )

    return algorithms
