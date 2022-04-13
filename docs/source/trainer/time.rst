|:hourglass:| Time
==================

We use the |Time| class to represent and track time throughout
the training loop. We track several time-related quantities 
(epochs, batches, samples, and tokens) throughout training and
represent them as elements of the |TimeUnit| enum class. Values
can be provided as a string:

.. csv-table::
   :header: "Unit", "Suffix", "Example", "Enum"
   :widths: 15, 10, 15, 30
   :width: 0.5
   :align: left

   "Epochs", ``"ep"``, ``"10ep"``, :attr:`.TimeUnit.EPOCH`
   "Batches", ``"ba"``, ``"100ba"``, :attr:`.TimeUnit.BATCH`
   "Samples", ``"sp"``, ``"2048sp"``, :attr:`.TimeUnit.SAMPLE`
   "Tokens", ``"tok"``, ``"93874tok"``, :attr:`.TimeUnit.TOKEN`
   "Duration", ``"dur"``, ``"0.7dur"``, :attr:`.TimeUnit.DURATION`

Duration is defined as a multiplier of the ``max_duration``.

These above string inputs are valid when an argument accepts the |Time|
type. There are some exceptions -- for example ``dur`` is not valid when
setting ``max_duration`` as that is circular.

Users can also specify milestones for objects such as learning rate schedulers
in units of ``duration``, e.g. ``0.1dur``. This makes it easy to build recipes
such as “decay the learning rate 10% into training”.

.. warning::

    For ``dur`` arguments, we keep the same units as used in ``max_duration``,
    and round down. For example, if ``max_duration = "7ep"`` and  ``warmup = "0.2dur"``,
    then warmup will be converted to ``floor(7 * 0.2) = 1 epoch``.


We also support arithmetic between instances that share the same units. For more information,
see the documentation for |Time|.

Tracking Time
-------------
The trainer has a :class:`.Timer` object stored in :attr:`.State.timer` that
measures progress in all the time formats above. :attr:`.State.timer` can be
read by algorithms and callbacks to trigger behavior at different times
during training. This feature allows algorithms to specify time in whatever unit
is most useful -- e.g. an algorithm could activate once every *n* batches or
during the last 20% of training.

When the trainer's timer unit is specified in terms of samples or tokens,
the timer increments time in response to the data being consumed. As each 
batch of data is read, the timer accumulates the total number of samples 
and/or tokens consumed.

By default, we attempt to infer the number of samples based on the batch type:

- If :class:`torch.Tensor`, the size of its first dimension is used.
- If ``list`` or ``tuple``, the size of its first dimension is used. As such, all elements must have the same first dimension size.
- If ``dict``, the size of its first dimension is used. As such, all elements must have the same first dimension size


Users can supply their own ``get_num_samples_in_batch`` method to the trainer
via the :class:`.DataSpec` for more complicated datasets:

.. code:: python

    from composer.core import DataSpec
    from composer import Trainer

    def my_num_samples(batch: dict) -> int:
        return batch['image1'].shape[0] + batch['image2'].shape[0]


    data_spec = DataSpec(
        dataloader=my_train_dataloader,
        get_num_samples_in_batch=my_num_samples,
    )

    trainer = Trainer(
        model=model,
        train_dataloader=data_spec,
    )


To track tokens properly, users will need to supply the ``get_num_tokens_in_batch``
function to the Trainer; otherwise, tokens will not be tracked.

Samples Per Epoch
-----------------

To convert between samples and epochs, we infer the number of samples per epoch
from ``len(dataloader.dataset)`` if the property is available. If not, we assume
the dataset is unsized.

``num_samples`` can also be provided directly to the :class:`.DataSpec` to override 
this default behavior.

.. code:: python

    from composer.core import DataSpec
    from composer import Trainer

    trainer = Trainer(
        model=model,
        train_dataloader=DataSpec(
            dataloader=my_train_dataloader,
            num_samples=1028428,
        )
    )

..
    TODO: discuss how to handle `drop_last`
    TODO: warn users against converting between time units

.. |Timer| replace:: :class:`.Timer`
.. |Time| replace:: :class:`.Time`
.. |TimeUnit| replace:: :class:`.TimeUnit`