composer.Algorithm
==================

.. currentmodule:: composer

We describe programmatic modifications to the model or training process as "algorithms."
Examples include :py:class:`smoothing the labels <composer.algorithms.label_smoothing.LabelSmoothing>`
and adding :py:class:`Squeeze-and-Excitation <composer.algorithms.squeeze_excite.SqueezeExcite>` blocks,
among many others.

Algorithms are implemented in both a standalone functional form (see :doc:`/walkthrough/functional`)
and as subclasses of :class:`Algorithm` for integration in the Composer :class:`Trainer`.
The former are easier to integrate piecemeal into an existing codebase.
The latter are easier to compose together, since they all have the same public interface
and work automatically with the Composer :py:class:`~composer.trainer.Trainer`.

For ease of composability, algorithms in our Trainer are based on the two-way callbacks concept from
`Howard et al., 2020 <https://arxiv.org/abs/2002.04688>`_. Each algorithm implements two methods:

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

For reference, available events include:

.. include:: event_table.rst

For more information about events, see :doc:`event`.


For a complete list of Algorithms, see:

.. _master-algo-list:


API Reference
*************

For a list of algorithms available in composer, see the :mod:`API Reference <composer.algorithms>`.

