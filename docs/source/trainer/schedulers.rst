|:chart_with_downwards_trend:| Schedulers
=========================================

The :class:`.Trainer` supports both PyTorch :mod:`torch.optim.lr_scheduler` schedulers
as well as our own schedulers, which take advantage of the :class:`.Time` representation.

For pytorch schedulers, we step every epoch by default. To instead step every batch, set
``use_stepwise_scheduler=True``:

.. code:: python

    from torch.optim.lr_scheduler import CosineAnnealingLR
    from composer import Trainer

    trainer = Trainer(
        schedulers=CosineAnnealingLR(...),
        use_stepwise_scheduler=True,
    )

.. note::

    If setting ``use_stepwise_schedulers`` to ``True``, remember to specify the
    arguments to your pytorch scheduler in units of batches, not epochs.

Our experiments have shown better accuracy using stepwise schedulers, and so
is the recommended setting in most cases.

Composer Schedulers
-------------------

Our schedulers differ from the pytorch schedulers in two ways:

- Time parameters can be provided in different units:
  samples (``"sp"``), tokens (``"tok"``), batches (``"ba"``), epochs (``"ep"``),
  and duration (``"dur"``). See :doc:`Time</trainer/time>`.
- our schedulers are functions, not classes. They return a multiplier to apply to
  the optimizer's learning rate, given the current trainer state, and optionally
  a "scale schedule ratio" (ssr).

For example, the following are equivalent:

.. code:: python

    from composer.optim.scheduler import step_scheduler

    # assume trainer has max_duration=50 epochs

    scheduler1 = lambda state: step_scheduler(state, step_size=['5ep', '25ep'])
    scheduler2 = lambda state: step_scheduler(state, step_size=['0.1dur', '0.5dur'])

    trainer = Trainer(
        ...
        schedulers = scheduler1
    )

These schedulers typically read the ``state.timer`` to determine the trainer's progress
and return a learning rate multipler. Inside the Trainer, we convert these to
:class:`torch.optim.lr_scheduler.LabmdaLR` schedulers. By default, our schedulers
have ``use_stepwise_scheduler=True``.

Below are the supported schedulers found at :mod:`composer.optim.scheduler`.

.. currentmodule:: composer.optim.scheduler
.. autosummary::
    :nosignatures:

    StepScheduler
    MultiStepScheduler
    MultiStepWithWarmupScheduler
    ConstantScheduler
    LinearScheduler
    LinearWithWarmupScheduler
    ExponentialScheduler
    CosineAnnealingScheduler
    CosineAnnealingWithWarmupScheduler
    CosineAnnealingWarmRestartsScheduler
    PolynomialScheduler

Scale Schedule Ratio
--------------------

The Scale Schedule Ratio (SSR) scales the learning rate schedule by a factor, and
is a powerful way to tradeoff training time and quality. ``ssr`` is an argument to
the :class:`.Trainer`.

.. TODO: add more here / add a figure

