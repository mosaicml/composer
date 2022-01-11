composer.Algorithm
==================

.. currentmodule:: composer

Algorithms are implemented in both a standalone functional form (see :doc:`../functional`) and as subclasses of :class:`Algorithm` for integration in the MosaicML :class:`Trainer`. This section describes the latter form.

For ease of composability, algorithms in our Trainer are based on the two-way callbacks concept from `Howard et al., 2020 <https://arxiv.org/abs/2002.04688>`_. Each algorithm implements two methods:

* :meth:`Algorithm.match`: returns ``True`` if the algorithm should be run given the current
  :class:`State` and :class:`Event`.
* :meth:`Algorithm.apply`: performs an in-place modification of the given
  :class:`State`

For example, a simple algorithm that shortens training:

.. code-block:: python

    from composer import Algorithm, State, Event, Logger

    class ShortenTraining(Algorithm):

        def match(self, state: State, event: Event, logger: Logger) -> bool:
            return event == Event.TRAINING_START

        def apply(self, state: State, event: Event, logger: Logger):
            state.max_duration /= 2  # cut training time in half

For a complete list of algorithms, see :doc:`/algorithms`.

For reference, available events include:

.. include:: event_table.rst

For more information about events, see :doc:`event`.

.. autoclass:: Algorithm
    :members:
