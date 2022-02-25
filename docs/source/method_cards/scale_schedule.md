# 🗜️ Scale Schedule

![scale_schedule.png](https://storage.googleapis.com/docs.mosaicml.com/images/methods/scale_schedule.png)

Tags: `Best Practice`, `Speedup`

## TL;DR

Scale Schedule changes the number of training steps by a dilation factor and dilating learning rate changes accordingly. Doing so varies the training budget, making it possible to explore tradeoffs between cost (measured in time or money) and the quality of the final model.

## Attribution

The number of training steps to perform is an important hyperparameter to tune when developing a model. This technique appears implicitly throughout the deep learning literature. One example of a systematic study of this approach is the *scan-SGD* technique in *[How Important is Importance Sampling for Deep Budgeted Training*](https://openreview.net/forum?id=TqQ0oOzJlai) by Eric Arazo, Diego Ortega, Paul Albert, Noel O'Connor, and Kevin McGuinness. Posted to OpenReview in 2020.

## Hyperparameters

- `ratio` - The ratio of the scaled learning rate schedule to the full learning rate schedule. For example, a ratio of 0.8 would train for 80% as many steps as the original schedule.
- `method` - Whether to scale the number of training `epochs` performed or the number of training `samples`used. `samples` is not currently supported.

## Example Effects

Changing the length of training will affect the final accuracy of the model. For example, training ResNet-50 on ImageNet for the standard schedule in the `composer` library leads to final validation accuracy of 76.6%, while using scale schedule with a ratio of 0.5 leads to final validation accuracy of 75.6%. Training for longer can lead to diminishing returns or even overfitting and worse validation accuracy. In general, the cost of training is proportional to the length of training when using scale schedule (assuming all other techniques, such as progressive resizing, have their schedules scaled accordingly).

## Implementation Details

Scale schedule is only called once at the beginning of training, and it alters the `max_epochs` stored in the `composer.State()` object.

The state update looks something like:

```python
new_max_epochs = int(state.max_epochs * ratio)
state.max_epochs = new_max_epochs
```

Scale schedule is currently only possible with the following learning rate schedulers:

 `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, `CosineAnnealingWarmRestarts`,

Scale schedule does *not* scale the warmup, if it is included at the beginning of the learning rate schedule.

If other methods specify absolute steps during training, these are *not* affected by scale schedule.

## Suggested Hyperparameters

The default scale schedule ratio is 1.0. For a standard maximum number of epochs (these will differ depending on the task), scaling down the learning rate schedule will lead to a monotonic decrease in accuracy. Increasing the scale schedule ratio will often improve the accuracy up to a plateau, although this leads to longer training time and added cost.

## Composability

As general rule, scale schedule can be applied in conjunction with any method. If other methods also perform actions according to a schedule, it is important to modify their schedules to coincide with the altered number of epochs.

---

## Code

```{eval-rst}
.. autoclass:: composer.algorithms.scale_schedule.scale_schedule.ScaleSchedule
    :members: match, apply
    :noindex:

.. autofunction:: composer.trainer._scale_schedule.scale_scheduler
    :noindex:
```