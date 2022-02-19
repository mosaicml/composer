Time
====

We use a |Time| class to represent and track time throughout
the training loop. We track several quantities (epochs, batches,
samples, and tokens) throughout training. Values
can be provided as a string:

- epochs: `"10ep"`
- batches: `"100ba"`
- samples: `"2048sp"`
- tokens: `"10242948tok"`
- duration: `0.7dur` (treated as a fraction of the trainer's ``max_duration``)

These above string inputs are valid when an argument accepts the |Time|
type. There are some exceptions -- for example ``dur`` is not valid when setting
``max_duration``.

Users can also specify milestones for objects such as Llearning rate schedulers
in units of ``duration``, such as ``0.1dur``. This makes it easy to build recipes
such as “a linear LR warmup over the first 1% of training”.

.. warning::

    For `dur` arguments, we keep the same units as that provided to ``max_duration``,
    and round down. For example, if ``max_duration = "7ep"``


We also support arithmetic between instances that share the same units. For more information,
see the documentation for |Time|.

The trainer had a :class:`.Timer` object stored in ``state.timer`` that
accurately measures all the time formats as training progresses. Training
is stopped once the timer's clock has exceeded ``max_duration``.

Time Conversions
----------------




between these units, provided that the trainer is supplied with the proper conversion factors.
For example, to convert between epochs and batches, the supplied `train_dataloader` must either
implement the `__len__` method,







.. code:: yaml

   schedulers:
     - warmup:
         warmup_method: linear
         warmup_iters: 0.01dur
         interval: batch

Before training begins, the Trainer statically converts units of ``dur``
to the granularity specified in ``interval``. At MosaicML we have found
in our research that fine-grained LR scheduling leads to better model
training (LINK HERE), so we step schedulers at the ``batch`` granularity
by default.

Also, the Trainer’s ``state.timer`` object can be read by algorithms and
callbacks. This allows algorithms to trigger different behavior at
different times during training, such as once-every-N batches, or during
the last 20% of training, etc.

How does it work?
-----------------

The Composer Trainer increments time as data is consumed. As each batch
of data is read, Composer sums the total number of samples and tokens
across all devices and accumulates the value into ``state.timer``.
Default methods are provided, but ``get_num_samples_in_batch`` and
``get_num_tokens_in_batch`` can also be replaced with custom functions.
Batches and epochs are similarly incremented.


.. |Timer| replace:: :class:`.Timer`
.. |Time| replace:: :class:`.Time`
.. |TimeUnit| replace:: :class:`.TimeUnit`