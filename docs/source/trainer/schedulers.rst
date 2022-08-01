|:chart_with_downwards_trend:| Schedulers
=========================================

The :class:`.Trainer` supports both PyTorch :mod:`torch.optim.lr_scheduler` schedulers
as well as our own schedulers, which take advantage of the :class:`.Time` representation.

For PyTorch schedulers, we step every epoch by default. To instead step every batch, set
``step_schedulers_every_batch=True``:

.. testcode::

    from composer import Trainer
    from torch.optim.lr_scheduler import CosineAnnealingLR

    trainer = Trainer(
        ...,
        schedulers=CosineAnnealingLR(optimizer, T_max=2),
        step_schedulers_every_batch=True,
    )

.. note::

    If setting ``step_schedulers_every_batch`` to ``True``, remember to specify the
    arguments to your pytorch scheduler in units of batches, not epochs.

Our experiments have shown better accuracy using stepwise schedulers, so
it is the recommended setting in most cases.

.. _Composer Schedulers:

Composer Schedulers
-------------------

Our schedulers take advantage of our :doc:`Time</trainer/time>` abstraction
to provide easier ways to set time. Time parameters can be provided in different units:
samples (``"sp"``), tokens (``"tok"``), batches (``"ba"``), epochs (``"ep"``),
and duration (``"dur"``). See :doc:`Time</trainer/time>`.

For example, the below would step the learning rate at 30%, 50%, and
90% of the way through training:


.. testcode::

    from composer import Trainer
    from composer.optim.scheduler import MultiStepScheduler

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        max_duration='90ep',
        schedulers=MultiStepScheduler(
            milestones=['0.3dur', '0.5dur', '0.9dur'],
            gamma=0.1
        ))

These schedulers typically read the ``state.timestamp`` to determine the trainer's progress
and return a learning rate multipler. Inside the Trainer, we convert these to
:class:`~torch.optim.lr_scheduler.LabmdaLR` schedulers. By default, our schedulers
are stepped at every batch.

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
    PolynomialWithWarmupScheduler

.. note::

    Compared to PyTorch schedulers, :class:`.ComposerScheduler` need not be provided
    an optimizer directly. The trainer will handle binding the optimizer when
    it compiles the scheduler later.

.. _Scale Schedule Ratio:

Scale Schedule Ratio
--------------------

The Scale Schedule Ratio (SSR) scales the learning rate schedule by some factor and
is a powerful way to tradeoff training time and quality. ``scale_schedule_ratio``
is an argument to the :class:`.Trainer`.

Scale Schedule changes the training duration by a scaling factor and
scales the learning rate scheduler accordingly. This serves to vary the
training budget, making it possible to explore tradeoffs between cost
(measured in time or money) and model quality.

For example, the code below will scale the training time by half
(to 10 epochs) and also scale the learning rate schedule.

.. testcode::

    from composer import Trainer
    from composer.optim.scheduler import MultiStepScheduler

    trainer = Trainer(
        ...,
        max_duration="20ep",
        schedulers=MultiStepScheduler(milestones=["10ep", "16ep"]),
        scale_schedule_ratio=0.5,
    )

    # or equivalently, with default SSR=1.0:

    trainer = Trainer(
        ...,
        max_duration="10ep",
        schedulers=MultiStepScheduler(milestones=["5ep", "8ep"])
    )

Importantly, for our schedulers that have warmup, the warmup
period is *not* scaled by default. For example, if we apply
``scale_schedule_ratio=0.5`` to:

.. testcode::

    from composer.optim.scheduler import MultiStepWithWarmupScheduler

    scheduler = MultiStepWithWarmupScheduler(
        milestones=["10ep", "20ep"],
        t_warmup="4ep",
    )

The resulting scheduler would warmup for 4 epochs and then
have step milestones at 5 epochs and 10 epochs.

To scale the warmup period as well, set ``scale_warmup=True``. For example:

.. testcode::

    from composer.optim.scheduler import MultiStepWithWarmupScheduler

    scheduler = MultiStepWithWarmupScheduler(
        milestones=["10ep", "20ep"],
        t_warmup="4ep",
        scale_warmup=True,
    )

With ``scale_schedule_ratio=0.5``, this scheduler will warmup for 2 epochs,
then step on 5 and 10 epochs.
