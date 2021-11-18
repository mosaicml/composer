composer.Callback
=================

.. currentmodule:: composer

Callbacks are run at each given :class:`Event`, and are used to for
non-essential recording functions such as logging or timing.

Callbacks differ from :class:`Algorithm` in that
they do not modify the training of the model.
By convention, callbacks should not modify the :class:`State`.


Each callback inherits from the :class:`Callback` base class,
and overrides the :meth:`~Callback.run_event` method.

For example:

.. code-block:: python

    from composer import Callback

    class MyCallback(Callback)

        def _run_event(self, event: Event, state: State, logger: Logger):
            if event == Event.EPOCH_START:
                print(f'Epoch {state.epoch}/{state.max_epochs}')

.. note::

    To use Composer's built in callbacks, see :doc:`/callbacks`.

.. autosummary::
    :recursive:
    :toctree: generated
    :nosignatures:

    ~composer.Callback
    ~composer.core.callback.RankZeroCallback
