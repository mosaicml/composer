# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Training Loop Events."""
from composer.utils import StringEnum

__all__ = ['Event']


class Event(StringEnum):
    """Enum to represent training loop events.

    Events mark specific points in the training loop where an :class:`~.core.Algorithm` and :class:`~.core.Callback`
    can run.

    The following pseudocode shows where each event fires in the training loop:

    .. code-block:: python

        # <INIT>
        # <AFTER_LOAD>
        # <FIT_START>
        for epoch in range(NUM_EPOCHS):
            # <EPOCH_START>
            while True:
                # <BEFORE_DATALOADER>
                batch = next(dataloader)
                if batch is None:
                    break
                # <AFTER_DATALOADER>

                # <BATCH_START>

                # <BEFORE_TRAIN_BATCH>

                for microbatch in batch.split(grad_accum):

                    # <BEFORE_FORWARD>
                    outputs = model(batch)
                    # <AFTER_FORWARD>

                    # <BEFORE_LOSS>
                    loss = model.loss(outputs, batch)
                    # <AFTER_LOSS>

                    # <BEFORE_BACKWARD>
                    loss.backward()
                    # <AFTER_BACKWARD>

                # Un-scale and clip gradients

                # <AFTER_TRAIN_BATCH>
                optimizer.step()

                # <BATCH_END>

                if should_eval(batch=True):
                    for eval_dataloader in eval_dataloaders:
                        # <EVAL_START>
                        for batch in eval_dataloader:
                            # <EVAL_BATCH_START>
                            # <EVAL_BEFORE_FORWARD>
                            outputs, targets = model(batch)
                            # <EVAL_AFTER_FORWARD>
                            metrics.update(outputs, targets)
                            # <EVAL_BATCH_END>
                        # <EVAL_END>

                # <BATCH_CHECKPOINT>
            # <EPOCH_END>

            if should_eval(batch=False):
                for eval_dataloader in eval_dataloaders:
                    # <EVAL_START>
                    for batch in eval_dataloader:
                        # <EVAL_BATCH_START>
                        # <EVAL_BEFORE_FORWARD>
                        outputs, targets = model(batch)
                        # <EVAL_AFTER_FORWARD>
                        metrics.update(outputs, targets)
                        # <EVAL_BATCH_END>
                    # <EVAL_END>

            # <EPOCH_CHECKPOINT>
        # <FIT_END>

    Attributes:
        INIT: Invoked in the constructor of :class:`~.trainer.Trainer`. Model surgery (see
            :mod:`~composer.utils.module_surgery`) typically occurs here.
        AFTER_LOAD: Immediately after checkpoint is loaded in constructor of :class:`~.trainer.Trainer`.
        FIT_START: Invoked at the beginning of each call to :meth:`.Trainer.fit`. Dataset transformations typically
            occur here.
        EPOCH_START: Start of an epoch.
        BEFORE_DATALOADER: Immediately before the dataloader is called.
        AFTER_DATALOADER: Immediately after the dataloader is called.  Typically used for on-GPU dataloader transforms.
        BATCH_START: Start of a batch.
        BEFORE_TRAIN_BATCH: Before the forward-loss-backward computation for a training batch. When using gradient
            accumulation, this is still called only once.
        BEFORE_FORWARD: Before the call to ``model.forward()``.
            This is called multiple times per batch when using gradient accumulation.
        AFTER_FORWARD: After the call to ``model.forward()``.
            This is called multiple times per batch when using gradient accumulation.
        BEFORE_LOSS: Before the call to ``model.loss()``.
            This is called multiple times per batch when using gradient accumulation.
        AFTER_LOSS: After the call to ``model.loss()``.
            This is called multiple times per batch when using gradient accumulation.
        BEFORE_BACKWARD: Before the call to ``loss.backward()``.
            This is called multiple times per batch when using gradient accumulation.
        AFTER_BACKWARD: After the call to ``loss.backward()``.
            This is called multiple times per batch when using gradient accumulation.
        AFTER_TRAIN_BATCH: After the forward-loss-backward computation for a training batch. When using gradient
            accumulation, this event still fires only once.
        BATCH_END: End of a batch, which occurs after the optimizer step and any gradient scaling.
        BATCH_CHECKPOINT: After :attr:`.Event.BATCH_END` and any batch-wise evaluation. Saving checkpoints at this
            event allows the checkpoint saver to use the results from any batch-wise evaluation to determine whether
            a checkpoint should be saved.
        EPOCH_END: End of an epoch.
        EPOCH_CHECKPOINT: After :attr:`.Event.EPOCH_END` and any epoch-wise evaluation. Saving checkpoints at this
            event allows the checkpoint saver to use the results from any epoch-wise evaluation to determine whether
            a checkpointshould be saved.
        FIT_END: Invoked at the end of each call to :meth:`.Trainer.fit`. This event exists primarily for logging information
            and flushing callbacks. Algorithms should not transform the training state on this event, as any changes will not
            be preserved in checkpoints.

        EVAL_START: Start of evaluation through the validation dataset.
        EVAL_BATCH_START: Before the call to ``model.eval_forward(batch)``
        EVAL_BEFORE_FORWARD: Before the call to ``model.eval_forward(batch)``
        EVAL_AFTER_FORWARD: After the call to ``model.eval_forward(batch)``
        EVAL_BATCH_END: After the call to ``model.eval_forward(batch)``
        EVAL_END: End of evaluation through the validation dataset.
    """

    INIT = 'init'
    AFTER_LOAD = 'after_load'
    FIT_START = 'fit_start'

    EPOCH_START = 'epoch_start'

    BEFORE_DATALOADER = 'before_dataloader'
    AFTER_DATALOADER = 'after_dataloader'

    BATCH_START = 'batch_start'

    BEFORE_TRAIN_BATCH = 'before_train_batch'

    BEFORE_FORWARD = 'before_forward'
    AFTER_FORWARD = 'after_forward'

    BEFORE_LOSS = 'before_loss'
    AFTER_LOSS = 'after_loss'

    BEFORE_BACKWARD = 'before_backward'
    AFTER_BACKWARD = 'after_backward'

    AFTER_TRAIN_BATCH = 'after_train_batch'

    BATCH_END = 'batch_end'
    BATCH_CHECKPOINT = 'batch_checkpoint'

    EPOCH_END = 'epoch_end'
    EPOCH_CHECKPOINT = 'epoch_checkpoint'

    FIT_END = 'fit_end'

    EVAL_START = 'eval_start'
    EVAL_BATCH_START = 'eval_batch_start'
    EVAL_BEFORE_FORWARD = 'eval_before_forward'
    EVAL_AFTER_FORWARD = 'eval_after_forward'
    EVAL_BATCH_END = 'eval_batch_end'
    EVAL_END = 'eval_end'

    PREDICT_START = 'predict_start'
    PREDICT_BATCH_START = 'predict_batch_start'
    PREDICT_BEFORE_FORWARD = 'predict_before_forward'
    PREDICT_AFTER_FORWARD = 'predict_after_forward'
    PREDICT_BATCH_END = 'predict_batch_end'
    PREDICT_END = 'predict_end'

    @property
    def is_before_event(self) -> bool:
        """Whether the event is an "before" event.

        An "before" event (e.g., :attr:`~Event.BEFORE_LOSS`) has a corresponding "after" event
        (.e.g., :attr:`~Event.AFTER_LOSS`).
        """
        return self in _BEFORE_EVENTS

    @property
    def is_after_event(self) -> bool:
        """Whether the event is an "after" event.

        An "after" event (e.g., :attr:`~Event.AFTER_LOSS`) has a corresponding "before" event
        (.e.g., :attr:`~Event.BEFORE_LOSS`).
        """
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
        name = name.replace('before_', '')
        name = name.replace('after_', '')
        name = name.replace('_start', '')
        name = name.replace('_end', '')
        return name

    @property
    def is_predict(self) -> bool:
        """Whether the event is during the predict loop."""
        return self.value.startswith('predict')

    @property
    def is_eval(self) -> bool:
        """Whether the event is during the eval loop."""
        return self.value.startswith('eval')


_BEFORE_EVENTS = (Event.FIT_START, Event.EPOCH_START, Event.BEFORE_DATALOADER, Event.BATCH_START,
                  Event.BEFORE_TRAIN_BATCH, Event.BEFORE_FORWARD, Event.BEFORE_LOSS, Event.BEFORE_BACKWARD,
                  Event.EVAL_START, Event.EVAL_BATCH_START, Event.EVAL_BEFORE_FORWARD, Event.PREDICT_START,
                  Event.PREDICT_BATCH_START, Event.PREDICT_BEFORE_FORWARD)
_AFTER_EVENTS = (Event.EPOCH_END, Event.BATCH_END, Event.AFTER_DATALOADER, Event.AFTER_TRAIN_BATCH, Event.AFTER_FORWARD,
                 Event.AFTER_LOSS, Event.AFTER_BACKWARD, Event.EVAL_END, Event.EVAL_BATCH_END, Event.EVAL_AFTER_FORWARD,
                 Event.FIT_END, Event.PREDICT_END, Event.PREDICT_BATCH_END, Event.PREDICT_AFTER_FORWARD)
