|:gear:| Using the Trainer
==========================

.. _using_composer_trainer:

The Composer :class:`.Trainer` implements a highly-optimized PyTorch training loop for neural networks. Using the trainer gives you several superpowers:

-  Easily insert our library of efficiency methods into the
   trainer loop and compose them to train better models faster.
-  Strong optimized baseline implementations to kick off your deep
   learning work, with reproducible results in time-to-train and
   accuracy.
-  Integration with your favorite model hubs:
   🤗 `Transformers`_, `TIMM`_, and `torchvision`_.
-  Iterate faster! We take care of performance and efficiency.

.. note::

    We use the two-way callback system developed by (`Howard et al,
    2020 <https://arxiv.org/abs/2002.04688>`__) to flexibly add the logic of
    our speedup methods during training.


Below are simple examples for getting started with the Composer Trainer
along with code snippets for more advanced usage such as using speedup
methods, checkpointing, and distributed training.

Getting Started
---------------

Create a model class that meets the :class:`.ComposerModel` interface,
minimally implementing the following methods:

-  ``def forward(batch) -> outputs`` : computes the forward pass based
   on the ``batch`` returned from the dataloader.
-  ``def loss(batch, outputs)``: returns the loss based on the
   ``outputs`` from the forward pass and the dataloader.

For more information, see the :doc:`ComposerModel</composer_model>` guide.

A minimal example of a ResNet-18 model is shown here:

.. testcode::

   import torchvision
   import torch.nn.functional as F

   from composer.models import ComposerModel

   class ResNet18(ComposerModel):

       def __init__(self):
           super().__init__()
           self.model = torchvision.models.resnet18()

       def forward(self, batch):
           inputs, _ = batch
           return self.model(inputs)

       def loss(self, outputs, batch):
           _, targets = batch
           return F.cross_entropy(outputs, targets)

Then, the model can be passed to the trainer with the relevant torch
objects.

.. code:: python

   import torch

   trainer = Trainer(model=ResNet18(),
                     train_dataloader=train_dataloader,
                     eval_dataloader=eval_dataloader,
                     optimizers=torch.optim.Adam(lr=0.01),
                     max_duration=10,  # epochs
                     device='gpu')

   trainer.fit()

In the background, we automatically add the :class:`.ProgressBarLogger` to log
training progress to the console.


A few tips and tricks for using our Trainer:

-  For time-related inputs, such as the ``max_duration`` above, we
   support both an integer (which we assume is epochs), or a string. The
   string can have a suffix of ``"ep"`` (epochs), ``"ba"`` (batches), or
   ``"dur"`` (full training duration), among other options.
   For example, ``"10ba"`` means 10 minibatches or steps, and ``"10ep"``
   means 10 epochs. See: :class:`.Time` for details.
-  If you are using gradient accumulation, the ``batch_size`` in your
   dataloaders should be the per-device macrobatch size — the batch size of your
   optimization update. For example, with ``grad_accum=2`` and
   ``batch_size=2048``, the trainer runs through two microbatches of 1024
   each, then performs a gradient update step.
-  At any time, most of the relevant quantities for debugging are
   centralized into one variable: :class:`.State`.
-  We have an abstraction for tracking :class:`.Time`, see the
   :doc:`Time<time>` guide.

For a full list of Trainer options, see :class:`.Trainer`. Below, we
illustrate some example use cases.

Training Loop
~~~~~~~~~~~~~

Behind the scenes, our trainer handles much of the engineering for
distributed training, gradient accumulation, device movement, gradient
scaling, and others. The *pseudocode* for our trainer loop as it
interacts with the :class:`.ComposerModel` is as follows:

.. code:: python

   # training loop
   for batch in train_dataloader:

       outputs = model.forward(batch)
       loss = model.loss(outputs, batch)

       loss.backward()
       optimizer.step()

   # eval loop
   for batch in eval_dataloader:
       outputs, targets = model.validate(batch)
       metrics.update(outputs, target)

For the actual code, see the :meth:`.Trainer.fit` and :meth:`.Trainer.eval` methods.

Quick Tour
----------

Below is a quick tour of various elements with code snippets for your reference. See the more
detailed sections in the navigation menu.

Events & State
~~~~~~~~~~~~~~

The core principle of the Composer trainer is to make it easy to inject
custom logic to run at various points in the training loop. To do this,
we have events that run before and after each of the lines above, e.g.:

.. code:: python

   engine.run_event("before_forward")
   outputs = model.forward(batch)
   engine.run_event("after_forward")

Algorithms and callbacks (see below) register themselves to run on one
or more events.

We also maintain a :class:`.State` which stores the trainer's state, such as
the model, optimizers, dataloader, current batch, etc (see
:class:`.State`). This allows algorithms to modify the state at the
various events above.

.. seealso::

    :doc:`events` and :class:`.State`

Algorithms
~~~~~~~~~~

The Composer trainer is designed to easily apply our library of
algorithms to both train more efficiently and build better models. These
can be enabled by passing the appropriate algorithm class to the ``algorithms``
argument.

.. testcode::

    from composer import Trainer
    from composer.algorithms import LayerFreezing, MixUp

    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      eval_dataloader=eval_dataloader,
                      max_duration='2ep',
                      algorithms=[
                        LayerFreezing(freeze_start=0.5, freeze_level=0.1),
                        MixUp(alpha=0.1),
                      ])

    # the algorithms will automatically be applied during the appropriate
    # points of the training loop
    trainer.fit()

.. testcleanup::

    trainer.engine.close()

We handle inserting those algorithms into the training loop and in the
right order.

.. seealso::

    Our :doc:`algorithms` guide, and the individual
    :doc:`/method_cards/methods_overview` for each algorithm.


Optimizers & Schedulers
~~~~~~~~~~~~~~~~~~~~~~~

You can easily specify which optimizer and learning rate scheduler to
use during training. Composer supports both PyTorch schedulers as
well as Composer's custom schedulers.

.. testcode::

   from composer import Trainer
   from composer.models import ComposerResNet
   from torch.optim import SGD
   from torch.optim.lr_scheduler import LinearLR

   model = ComposerResNet(model_name="resnet50", num_classes=1000)
   optimizer = SGD(model.parameters(), lr=0.1)
   scheduler = LinearLR(optimizer)

   trainer = Trainer(model=model,
                     train_dataloader=train_dataloader,
                     eval_dataloader=eval_dataloader,
                     max_duration='90ep',
                     optimizers=optimizer,
                     schedulers=scheduler)

Composer's own custom schedulers are versions that support the
:class:`.Time` abstraction. Time related inputs such as ``step``
or ``T_max`` can be provided in many units, from epochs (``"10ep"``)
to batches (``"2048ba"``) to duration (``"0.7dur"``).

For example, the below would step the learning rate at 30%, 50%, and
90% through training:


.. testcode::

    from composer import Trainer
    from composer.optim.scheduler import MultiStepScheduler

    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      max_duration='90ep',
                      schedulers=MultiStepScheduler(
                          milestones=['0.3dur', '0.5dur', '0.9dur'],
                          gamma=0.1
                      ))


See :doc:`schedulers` for details.


Training on GPU
~~~~~~~~~~~~~~~

Control which device you use for training with the ``device`` parameter,
and we will handle the data movement and other systems-related
engineering. We currently support the ``cpu`` and ``gpu`` devices.

.. testcode::

   from composer import Trainer

   trainer = Trainer(model=model,
                     train_dataloader=train_dataloader,
                     eval_dataloader=eval_dataloader,
                     max_duration='2ep',
                     device='cpu')

Distributed Training
~~~~~~~~~~~~~~~~~~~~

It's also simple to do data-parallel training on multiple GPUs. Composer
provides a launcher command that works with the trainer and handles all
the ``torch.distributed`` setup for you.

.. code:: python

   # run_trainer.py

   from composer import Trainer

   trainer = Trainer(model=model,
                     train_dataloader=train_dataloader,
                     eval_dataloader=eval_dataloader,
                     max_duration='160ep',
                     device='gpu')
   trainer.fit()

Access the Composer launcher via the ``composer`` command along with the
number of GPUs you'd like to use and your training script. Use
``composer --help`` to see a full list of configurable options.

.. code:: bash

   # run training on 8 GPUs
   $ composer -n 8 run_trainer.py

For multiple GPUs, the ``batch_size`` for each dataloader should be the
per-device batch size. For example, to use a batch size of 2048, with
data parallel across 8 GPUs, the dataloader should have ``batch_size=256``.


.. warning::

    For distributed training, your dataloader should use the
    :mod:`torch.utils.data.distributed.DistributedSampler`. If you
    are running multi-node, and each rank does not have a copy of the
    dataset, then a normal sampler can be used.


.. seealso::

    Our :doc:`distributed_training` guide and
    the :mod:`composer.utils.dist` module.


DeepSpeed Integration
~~~~~~~~~~~~~~~~~~~~~

Composer comes with DeepSpeed support, allowing you to leverage their
full set of features that makes it easier to train large models across
(1) any type of GPU and (2) multiple nodes. For more details on DeepSpeed,
see `their website <https://www.deepspeed.ai>`__.

To enable DeepSpeed, simply pass in a config as specified in the
DeepSpeed docs `here <https://www.deepspeed.ai/docs/config-json/>`__.

.. code:: python

   # run_trainer.py

   from composer import Trainer

   trainer = Trainer(model=model,
                     train_dataloader=train_dataloader,
                     eval_dataloader=eval_dataloader,
                     max_duration='160ep',
                     device='gpu',
                     deepspeed_config={
                         "train_batch_size": 2048,
                         "fp16": {"enabled": True},
                     })

Providing an empty dictionary to deepspeed is also valid. The deepspeed
defaults will be used and other fields (such as precision) inferred
from the trainer.

.. warning::

    The ``deepspeed_config`` must not conflict with any other parameters
    passed to the trainer.


Callbacks
~~~~~~~~~

You can insert arbitrary callbacks to be run at various points during
the training loop. The Composer library provides several useful
callbacks for things such as monitoring throughput and memory usage
during training, but you can also implement your own.

.. code:: python

   from composer import Trainer
   from composer.callbacks import SpeedMonitor

   # include a callback for tracking throughput/step during training
   trainer = Trainer(model=model,
                     train_dataloader=train_dataloader,
                     eval_dataloader=eval_dataloader,
                     max_duration='160ep',
                     device='gpu',
                     callbacks=[SpeedMonitor(window_size=100)])

.. seealso::

    The :doc:`callbacks` guide and :mod:`composer.callbacks`.


Numerics
~~~~~~~~

The trainer automatically handles multiple precision types, either as ``fp32`` or for GPUs,
``amp`` for automatic mixed precision, which is pytorch's built-in methods of training
in 16-bit floating point. For more details on ``amp``, see :mod:`torch.cuda.amp` and
the paper by `Micikevicius et al, 2018 <https://arxiv.org/abs/1710.03740>`__

We recommend using ``amp`` on GPUs to accelerate your training.

.. code:: python

   from composer import Trainer

   # use mixed precision during training
   trainer = Trainer(model=model,
                     train_dataloader=train_dataloader,
                     eval_dataloader=eval_dataloader,
                     max_duration='160ep',
                     device='gpu',
                     precision='amp')

Checkpointing
~~~~~~~~~~~~~

The Composer trainer makes it easy to (1) save checkpoints at various
points during training and (2) load them back to resume training later.

.. code:: python

   from composer import Trainer

   ### Saving checkpoints
   trainer = Trainer(model=model,
                     train_dataloader=train_dataloader,
                     eval_dataloader=eval_dataloader,
                     max_duration='160ep',
                     device='gpu',
                     # Checkpointing params
                     save_folder: 'checkpoints',
                     save_interval: '1ep')

   # will save checkpoints to the 'checkpoints' folder every epoch
   trainer.fit()

.. code:: python

   from composer import Trainer

   ### Loading checkpoints
   trainer = Trainer(model=model,
                     train_dataloader=train_dataloader,
                     eval_dataloader=eval_dataloader,
                     max_duration='160ep',
                     device='gpu',
                     # Checkpointing params
                     load_path: 'path/to/checkpoint/mosaic_states.pt')

   # will load the trainer state (including model weights) from the
   # load_path before resuming training
   trainer.fit()

.. seealso::

    The :doc:`checkpointing` guide.


This was just a quick tour of all the features within our trainer. Please see the other
guides and notebooks for more information.

Reproducibility
~~~~~~~~~~~~~~~

The random seed can be provided to the trainer directly, e.g.

.. testcode::

    from composer import Trainer

    trainer = Trainer(
        ...,
        seed=42,
    )

If no seed is provided, a random seed will be generated from system time.

Since the model and dataloaders are initialized outside of the Trainer, for complete
determinism, we recommend calling :func:`~composer.utils.reproducibility.seed_all` and/or
:func:`~composer.utils.reproducibility.configure_deterministic_mode` before creating any objects. For example:

.. testsetup::

    import functools
    import torch.nn
    import warnings

    warnings.filterwarnings(action="ignore", message="Deterministic mode is activated.")

    MyModel = functools.partial(SimpleBatchPairModel, num_channels, num_classes)

.. testcode::

   import torch.nn
   from composer.utils import reproducibility

   reproducibility.configure_deterministic_mode()
   reproducibility.seed_all(42)

   model = MyModel()

   def init_weights(m):
       if isinstance(m, torch.nn.Linear):
           torch.nn.init.xavier_uniform(m.weight)

   # model will now be deterministically initialized, since the seed is set.
   init_weights(model)
   trainer = Trainer(model=model, seed=42)

Note that the Trainer must still be seeded.

Annotated Trainer Loop
----------------------

Our :class:`.Trainer` code is meant to be easily readable and understood. In this section,
we walk you through the logic flow of the training loop code, from :meth:`.Trainer.fit`
down to the :meth:`~torch.tensor.backward` call.

In pseudocode, the trainer is organized as follows:

.. code:: python

    def fit(self):
        try:
            _train_loop()
        finally:  # clean up
            self.engine.close()


The method ``_train_loop()`` sets up the training, loads any
provided checkpoints, and then runs the training:


.. code:: python

    # pseudocode
    def _train_loop(self):

        # setup training
        # metrics, gradient scaling, etc.
        # if needed, load checkpoints

        while timer < max_duration:

            for batch in train_dataloader:

                # for grad accum, split the batch
                microbatches = split_batch(batch)

                """"
                Depending on the config, the _train_batch()
                is called with slightly different code.
                """
                if deepspeed_enabled:
                    loss = self._train_batch(microbatches)
                elif _use_closure():
                    """
                    Mixed precision and some optimizers
                    requiring passing a closure.
                    """"
                    loss = optimizer.step(
                        closure=self._train_batch(microbatches)
                    )
                else:
                    loss = self._train_batch(microbatches)
                    optimizer.step()

                if eval_on_batch:
                    eval()  # run validation

            if eval_on_epoch:
                eval()


Remaining are two methods: ``_train_batch`` and ``_train_batch_inner``.
For first decides whether to use the context manager for
:meth:`torch.nn.parallel.DistributedDataParallel.no_sync`, which
disables the gradient synchronization for distributed training.

The second carries out the iteration over the ``batch``, broken
into microbatches (for gradient accumulation). This last
method is where the forward and backward pass take place.

.. code:: python

    # pseudocode
    # handles precision, grad accum, etc.
    def _inner_train_batch(self, microbatches):
        for batch in microbatches:

            # run iteration
            outputs = model.forward(batch)
            loss = model.loss(outputs, batch)
            loss.backward()

.. _Transformers: https://huggingface.co/docs/transformers/index
.. _TIMM: https://fastai.github.io/timmdocs/
.. _torchvision: https://pytorch.org/vision/stable/models.html
