Welcome Tour
============

Welcome to the MosaicML *Composer* library! This guide will walk you through the basics of how the Composer trainer works, and how it interacts with our methods libary. This guide will assume you've already gone through the installation instructions.

Our First Method!
-----------------

We're going to explore how MixUp, a fairly simple algorithm, works. MixUp, introduced in `Zhang et al., 2017 <https://arxiv.org/abs/1710.09412>`_, is a regularization technique that tends to improve the accuracy of image categorization models.

MixUp operates by modifying the batches of data used to train the model; instead of training on individual samples, we train on convex combinations of samples. Thus, our implementation of the MixUp algorithm needs to be able to modify batches of training data after they are loaded from the dataloader and before they are passed into the forward pass of a model.

For more information on MixUp, see :doc:`/method_cards/mix_up` in our methods library.

So how can we use MixUp within a trainer?

A Simple Instrumented Trainer
-----------------------------

A very simple Pytorch training loop looks something like the following:

.. code-block:: python

    for epoch in range(NUM_EPOCHS):
        for inputs, targets in dataloader:
            outputs = model.forward(inputs)
            loss = model.loss(outputs, targets)
            loss.backward()

            optimizer.step()

MixUp needs to modify ``inputs`` and ``targets`` after they are loaded from the dataloader but before the inputs are passed to the forward pass of the model. So one possibility is we could make use of our functional API to modify our training loop:

.. code-block:: python

    from composer import functional as CF

    for epoch in range(NUM_EPOCHS):
        for inputs, targets in dataloader:

            inputs, targets = CF.mixup_batch(inputs, targets)

            outputs = model.forward(inputs)
            loss = model.loss(outputs, targets)
            loss.backward()

            optimizer.step()

This works, and is recommend if you want to quickly modify an existing training loop to use our implementation of MixUp! However, the goal of the Composer library is to be able to rapidly experiment with different combinations of algorithms. Our methods library contains over 20 different methods to experiment with, and it would be unwieldy to have to add conditional logic to the trainer for enabling/disabling each new method.

Introducing... Events, Engines, and State
-----------------------------------------

The core principle of the Composer trainer is to avoid the need to introduce algorithm-specific logic to the trainer by instead relying on callbacks tied to *events*. Events describe specific stages of the training lifecycle, such as ``BATCH_START`` and ``BEFORE_FORWARD``. We could instrument our training loop with events as follows:

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

Now we need a way to tie events to algorithms, so that we know which algorithms to run, and when to run them. This is the purpose of the :class:`~composer.core.Engine`. The :class:`~composer.core.Engine` is initialized with a list of algorithms to run, and provides a :meth:`composer.core.Engine.run_event` method that the trainer can call to execute algorithms for the given event. The :class:`~composer.core.Engine` also is responsible for handling potential conflicts between multiple algorithms.

One piece is missing. Algorithms are no longer running from within the body of the training loop, but they still need to be able to modify the training loop's state. For this, we introduce :class:`~composer.core.State`, which stores all objects relevant to training that algorithms need access to. The :class:`~composer.core.Engine` is initialized with a reference to the :class:`~composer.core.State` and passes it to algorithms when it invokes them.

Finally, to be compatible with the :class:`~composer.core.Engine`, algorithms need to implement two methods: :meth:`~composer.core.Algorithm.match` and :meth:`~composer.core.Algorithm.apply`. For MixUp, these methods can be very simple:

.. code-block:: python

    class MixUp(Algorithm):
        def match(self, event: Event, state: State) -> bool:
            """Determine whether the algorithm should run on a given event."""
            return event == "after_dataloader"

        def apply(self, event: Event, state: State, logger: Logger) -> None:
            """Run the algorithm by modifying the State."""
            state.inputs, state.targets = mixup_batch(state.inputs, state.targets)

Putting all the pieces together, our trainer looks something like this:

.. code-block:: python

    state = State(...)

    engine = Engine(state=state, algorithms=[MixUp()])

    engine.run_event("init")
    engine.run_event("fit_start")
    for epoch in range(NUM_EPOCHS):
        engine.run_event("epoch_start")
        for state.inputs, state.targets in dataloader:
            engine.run_event("after_dataloader")

            engine.run_event("batch_start")

            engine.run_event("before_forward")
            state.outputs = state.model.forward(state.inputs)
            engine.run_event("after_forward")

            engine.run_event("before_loss")
            state.loss = state.model.loss(state.outputs, state.targets)
            engine.run_event("after_loss")

            engine.run_event("before_backward")
            state.loss.backward()
            engine.run_event("after_backward")

            state.optimizers.step()
            state.schedulers.step()

            engine.run_event("batch_end")
        engine.run_event("epoch_end")

That's it! Our training loop is now taking full advantage of MixUp, and we can easily start using new algorithms! For more information on events, state, and engines, check out :doc:`/core/event`, :doc:`/core/state`, and :doc:`/core/engine`.

Next: The Composer Trainer
--------------------------

For advanced experimentation, we recommend using our provided trainer. Our trainer takes care of all the state management and event callbacks from above, and adds a bunch of advanced features, including hyperparameter management, gradient accumulation, and closure support. For more information, check out our trainer documentation at :doc:`/trainer`.