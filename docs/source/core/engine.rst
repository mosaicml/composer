composer.Engine
===============

The order of ``algorithms`` can matter significantly during composition. For example, the Selective Backprop algorithm runs during ``AFTER_DATALOADER`` event, and must run before any data augmentations. The :class:`composer.Engine` runs these re-ordering passes.

.. note::

    The design of the ``Engine`` will be changed in future releases to accomdate more complexity as we investigation the composition of algorithms.


Currently, the following passes are registered:

* **LIFO order for events**

   For events that follow the ``after_*`` and ``before_*`` pattern, the ordering of algorithms is reversed for the ``after_*`` events. For example, algorithms will run in a ``ABCD -> DCBA`` ordering before and after say, the loss computation.

   This allows algorithms to "clean up" their changes. e.g. Label smoothing will smooth the labels upon entry to the loss, and then restore the original unsmoothed labels upon exit.

* **Run Selective Backprop first**

   Selective backprop runs after the dataloader returns the batch, and executes an extra forward pass to rank and prune the examples in the batch by loss. To ensure a clean estimate of the example, Selective backprop should run before any other data augmentations during ``AFTER_DATALOADER`` (e.g. such as MixUp).

.. currentmodule:: composer.core

.. autoclass:: Engine
    :members:

Trace
~~~~~

``Trace`` records whether an algorithm ran at a particular step and ``Event`` combination, and also its order. These are logged with the key ``{algorithm_name}/{event}``.

For example, the algorithm ``Layer Freezing``, which runs at the end of every epoch, will emit a series of traces:

.. code-block::

   [STEP=0][layer_freezing/TRAINING_START=0]
   [STEP=1][layer_freezing/EPOCH_START=0]
   [STEP=1][layer_freezing/BATCH_START=0]
   ...
   [STEP=2][layer_freezing/BATCH_START=0]
   ...
   [STEP=3][layer_freezing/BATCH_START=0]
   ...
   [STEP=3][layer_freezing/EPOCH_END=1]  # <-- ran here!

.. autoclass:: Trace
    :members: