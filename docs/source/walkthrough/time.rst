composer.Time
=============

Composer includes a a training time tracking module to describe the current
progress in the training duration. Time is tracked in terms of epochs, batches,
samples, and tokens. Callbacks, algorithms, and schedulers can use the current training
time to fire at certain points in the training process.

The :class:`~composer.core.Timer` class tracks the total number of epochs, batches, samples, and tokens.
The trainer is responsible for updating the :class:`~composer.core.Timer` at the end of every epoch and batch.
There is only one instance of the :class:`~composer.core.Timer`, which is attached to the :class:`~composer.core.State`.

The :class:`~composer.core.Time` class represents static durations of training time or points in the
training process in terms of a specific :class:`~composer.core.TimeUnit` enum. The :class:`~composer.core.Time` class
supports comparisons, arithmetic, and conversions.

API Reference
*************

See :mod:`composer.core.time`.
