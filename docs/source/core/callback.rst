composer.Callback
=================

A callback is a piece of code that can run at any :doc:`Event </core/event>`. Callbacks differ from :doc:`algorithms </core/algorithm>` in that they are not expected to have a major impact on the training of a model; rather, they are generally used for non-essential recording functions such as logging or timing. By convention, a callback should not modify the :doc:`Event </core/state>`.

A complete list of callbacks can be found :doc:`here </trainer/callbacks>`.

.. currentmodule:: composer.core

.. py:class:: Callback

    Abstract base class for callbacks.
    
    Callbacks are similar to Algorithms, in that
    they are run on specific events. By convention, Callbacks should not
    modify :class:`State`.

    Each method name in corresponds to an :class:`Event` value.
    This class has a method for each value in the :class:`Event` enum, formatted in snake case. Subclasses of :class:`Callback` can override these methods to run in response to given :class:`Event` invocations.

    .. py:method:: <event_name>(state: State, logger: Logger)

        Called in response to the :class:`Event` corresponding to this method's name.

        :param state: The current state.
        :type state: State
        :param logger: A logger to use for logging callback-specific metrics.
        :type logger: Logger

