composer.Event
==============

.. currentmodule:: composer

Events represent specific points in the training loop where a :class:`Algorithm` and
:class:`Callback` can run.

.. note ::

    By convention, :class:`Callback` should not be modifying the state,
    and are used for non-essential reporting functions such as logging or timing.
    Methods that need to modify state should be :class:`Algorithm`.

Events List
~~~~~~~~~~~

Available events include:

.. include:: event_table.rst

API Reference
~~~~~~~~~~~~~

.. autoclass:: composer.Event
    :members:
