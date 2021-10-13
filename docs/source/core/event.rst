composer.Event
==============

Events represent specific points in the training loop where a :class:`~composer.core.algorithm.Algorithm` and
:class:`~composer.core.callback.Callback` can run.

.. note ::

    By convention, :class:`~composer.core.callback.Callback` should not be modifying the state,
    and are used for non-essential reporting functions such as logging or timing.
    Methods that need to modify state should be :class:`~composer.core.algorithm.Algorithm`.

Events List
~~~~~~~~~~~

Available events include:

.. include:: event_table.rst

Training Loop
~~~~~~~~~~~~~

.. include:: event_training_loop.rst

API Reference
~~~~~~~~~~~~~

.. autoclass:: composer.core.event.Event
    :members:
