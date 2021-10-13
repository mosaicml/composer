composer.Callback
=================

Callbacks are run at each given :class:`~composer.core.event.Event`, and are used to for
non-essential recording functions such as logging or timing.

Callbacks differ from :class:`~composer.core.algorithm.Algorithm` in that
they do not modify the training of the model.
By convention, callbacks should not modify the :class:`~composer.core.state.State`.


Each callback inherits from the :class:`~composer.core.callback.Callback` base class,
and overrides functions corresponding to the event.


For example:

.. code-block:: python

    from composer import Callback

    class MyCallback(Callback)

        def epoch_start(self, state: State, logger: Logger):
            print(f'Epoch {state.epoch}/{state.max_epochs}')

.. note::

    To use Composer's built in callbacks, see :doc:`/callbacks`.

.. autosummary::
    :recursive:
    :toctree: generated
    :nosignatures:

    ~composer.core.callback.Callback
    ~composer.core.callback.RankZeroCallback
