# Copyright 2021 MosaicML. All Rights Reserved.

"""Events represent specific points in the training loop where an :class:`~.core.Algorithm` and :class:`~.core.Callback`
can run."""
from composer.utils.string_enum import StringEnum

__all__ = ["Event"]


class Event(StringEnum):
    """Enum to represent events in the training loop.

    The following pseudocode shows where each event fires in the training loop:

    .. code-block:: python

        # <INIT>
        # <FIT_START>
        for epoch in range(NUM_EPOCHS):
            # <EPOCH_START>
            for inputs, targets in dataloader:
                # <AFTER_DATALOADER>

                # <BATCH_START>

                # <BEFORE_FORWARD>
                outputs = model.forward(inputs)
                # <AFTER_FORWARD>

                # <BEFORE_LOSS>
                loss = model.loss(outputs, targets)
                # <AFTER_LOSS>

                # <BEFORE_BACKWARD>
                loss.backward()
                # <AFTER_BACKWARD>

                optimizer.step()

                # <BATCH_END>
            # <EPOCH_END>




    Attributes:
        INIT: Invoked in the constructor of :class:`~.trainer.Trainer`. Model surgery (see
            :mod:`~composer.utils.module_surgery`) typically occurs here.
        FIT_START: Invoked at the beginning of each call to :meth:`.Trainer.fit`. Dataset transformations typically
            occur here.
        EPOCH_START: Start of an epoch.
        BATCH_START: Start of a batch.
        AFTER_DATALOADER: Immediately after the dataloader is called.  Typically used for on-GPU dataloader transforms.
        BEFORE_TRAIN_BATCH: Before the forward-loss-backward computation for a training batch. When using gradient
            accumulation, this is still called only once.
        BEFORE_FORWARD: Before the call to ``model.forward()``.
        AFTER_FORWARD: After the call to ``model.forward()``.
        BEFORE_LOSS: Before the call to ``model.loss()``.
        AFTER_LOSS: After the call to ``model.loss()``.
        BEFORE_BACKWARD: Before the call to ``loss.backward()``.
        AFTER_BACKWARD: After the call to ``loss.backward()``.
        AFTER_TRAIN_BATCH: After the forward-loss-backward computation for a training batch. When using gradient
            accumulation, this event still fires only once.
        BATCH_END: End of a batch, which occurs after the optimizer step and any gradient scaling.
        EPOCH_END: End of an epoch.

        EVAL_START: Start of evaluation through the validation dataset.
        EVAL_BATCH_START: Before the call to ``model.validate(batch)``
        EVAL_BEFORE_FORWARD: Before the call to ``model.validate(batch)``
        EVAL_AFTER_FORWARD: After the call to ``model.validate(batch)``
        EVAL_BATCH_END: After the call to ``model.validate(batch)``
        EVAL_END: End of evaluation through the validation dataset.
    """

    INIT = "init"
    FIT_START = "fit_start"

    EPOCH_START = "epoch_start"
    BATCH_START = "batch_start"

    AFTER_DATALOADER = "after_dataloader"

    BEFORE_TRAIN_BATCH = "before_train_batch"

    BEFORE_FORWARD = "before_forward"
    AFTER_FORWARD = "after_forward"

    BEFORE_LOSS = "before_loss"
    AFTER_LOSS = "after_loss"

    BEFORE_BACKWARD = "before_backward"
    AFTER_BACKWARD = "after_backward"

    AFTER_TRAIN_BATCH = "after_train_batch"

    BATCH_END = "batch_end"
    EPOCH_END = "epoch_end"

    EVAL_START = "eval_start"
    EVAL_BATCH_START = "eval_batch_start"
    EVAL_BEFORE_FORWARD = "eval_before_forward"
    EVAL_AFTER_FORWARD = "eval_after_forward"
    EVAL_BATCH_END = "eval_batch_end"
    EVAL_END = "eval_end"

    @property
    def is_before_event(self) -> bool:
        """Whether the event is a 'before_*' event (e.g., :attr:`~Event.BEFORE_LOSS`) and has a corresponding 'after_*'
        (.e.g., :attr:`~Event.AFTER_LOSS`)."""
        return self in _BEFORE_EVENTS

    @property
    def is_after_event(self) -> bool:
        """Whether the event is an 'after_*' event (e.g., :attr:`~Event.AFTER_LOSS`) and has a corresponding 'before_*'
        (.e.g., :attr:`~Event.BEFORE_LOSS`)."""
        return self in _AFTER_EVENTS

    @property
    def canonical_name(self) -> str:
        """The name of the event, without before/after markers.

        Events that have a corresponding "before" or "after" event share the same canonical name.

        Example:
            >>> Event.EPOCH_START.canonical_name
            'epoch'
            >>> Event.EPOCH_END.canonical_name
            'epoch'

        Returns:
            str: The canonical name of the event.
        """
        name: str = self.value
        name = name.replace("before_", "")
        name = name.replace("after_", "")
        name = name.replace("_start", "")
        name = name.replace("_end", "")
        return name


_BEFORE_EVENTS = (Event.EPOCH_START, Event.BATCH_START, Event.BEFORE_TRAIN_BATCH, Event.BEFORE_FORWARD,
                  Event.BEFORE_LOSS, Event.BEFORE_BACKWARD, Event.EVAL_START, Event.EVAL_BATCH_START,
                  Event.EVAL_BEFORE_FORWARD)
_AFTER_EVENTS = (Event.EPOCH_END, Event.BATCH_END, Event.AFTER_TRAIN_BATCH, Event.AFTER_FORWARD, Event.AFTER_LOSS,
                 Event.AFTER_BACKWARD, Event.EVAL_END, Event.EVAL_BATCH_END, Event.EVAL_AFTER_FORWARD)
