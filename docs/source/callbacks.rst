composer.callbacks
==================

``Callbacks`` are run at each given :doc:`Event </core/event>`, and should typically be non-essential recording functions such as logging or timing. By convention, ``Callbacks`` should not modify the state.

Each callback inherits from :class:`Callback` base class, and overrides functions corresponding to the event. For example:

.. code-block:: python

    from composer import Callback

    class MyCallback(Callback)

        def epoch_start(self, state: State, logger: Logger):
            print(f'Epoch {state.epoch}/{state.max_epochs}')

Within the `yahp` system, callbacks can be added with the `--callbacks` argparse flag:

.. code-block::

    python examples/run_mosaic_trainer.py -f my_model.yaml --callbacks lr_monitor grad_monitor

Available callbacks are:



.. currentmodule:: composer.callbacks

.. autosummary::
    :nosignatures:

    callback_hparams
    grad_monitor
    lr_monitor
    profiler
    speed_monitor
    timing_monitor

.. currentmodule:: composer.core
