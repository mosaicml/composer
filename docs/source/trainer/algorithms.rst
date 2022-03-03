|:robot:| Algorithms
====================

Under construction |:construction:|

Included in the Composer library is a suite of algorithmic speedup algorithms.
These modify the basic training procedure, and are intended to be *composed*
together to easily create a complex and hopefully more efficient training routine.
While other libraries may have implementations of some of these, the implementations
in Composer are specifically written to be combined with other methods.

Below is a brief overview of the algorithms currently in Composer.
For more detailed information about each algorithm, see the :doc:`method cards</method_cards/methods_overview>`,
also linked in the table. Each algorithm has a functional implementation intended
for use with your own training loop, and an implementation intended for use with
Composer's trainer.

.. csv-table::
    :header: "Name" , "tldr", "functional"
    :delim: |
    :widths: 30, 40, 30

    {% for name, data in metadata.items() %}
    {% if data.functional %}
    :doc:`{{ data.class_name }}</method_cards/{{name}}>` | {{ data.tldr }} | ``{{ data.functional }}``
    {% else %}
    :doc:`{{ data.class_name }}</method_cards/{{name}}>` | {{ data.tldr }} | {{ data.functional }}
    {% endif %}
    {% endfor %}

Functional API
--------------

The simplest way to use Composer's algorithms is through the functional API. Composer's
algorithms can be grouped into three broad classes:

- `data augmentations` add additional transforms to the training data.
- `model surgery` algorithms modify the network architecture.
- `training loop modifications` change various properties of the training loop.

Data augmentations can be inserted either into the dataloader as a transform, or after a
batch has been loaded depending on what the augmentation acts on. Here is an example of using
:class:`.RandAugment` with the functional API

.. code-block:: python

    import torch
    from torchvision import datasets, transforms

    from composer import functional as cf

    c10_transforms = transforms.Compose([cf.randaugment(), # <---- Add RandAugment
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    dataset = datasets.CIFAR10('../data',
                            train=True,
                            download=True,
                            transform=c10_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024)

Other data augmentations, such as :class:`.CutMix` act on a batch of inputs. These can be
inserted in the training loop after a batch is loaded from the dataloader as follows:

.. code-block:: python

    from composer import functional as CF

    cutmix_alpha = 1
    num_classes = 10
    for batch_idx, (data, target) in enumerate(dataloader):
    ### Insert CutMix here ###
    data = CF.cutmix(data, target, cutmix_alpha, num_classes)
    ### ------------------ ###
        optimizer.zero_grad()
    output = model(data)
    loss = loss(output, target)
    loss.backward()
    optimizer.step()

Model surgery algorithms make direct modifications to the network itself. Functionally,
these can be called as follows, using :class:`.BlurPool` as an example

.. code-block:: python

    import torchvision.models as models

    from composer import functional as cf

    model = models.resnet18()
    cf.apply_blurpool(model)

Each method card has a section describing how to use these methods in your own trainer loop.

Composer Trainer
----------------

To make full use of Composer, we recommend using our algorithms and trainer together.
Using algorithms with the trainer is simple, just pass a list of the algorithms you want to run
as the `algorithms` argument when initializing the trainer.
Composer will automatically run each algorithm at the appropriate time during training,
as well as handle any collisions and reorderings needed.

Here is an example of how to call trainer with a few algorithms:

.. code-block:: python

    from composer import Trainer
    from composer.algorithms.blurpool import BlurPool
    from composer.algorithms.channels_last import ChannelsLast

    channels_last = ChannelsLast()
    blurpool = BlurPool(replace_convs=True,
                                            replace_maxpools=True,
                                            blur_first=True)

    trainer = Trainer(model=model,
                    train_dataloader=train_dataloader,
                    eval_dataloader=test_dataloader,
                    max_duration='90ep',
                    device='gpu',
                    algorithms=[channels_last, blurpool],
                    validate_every_n_epochs=-1,
                    seed=42)

Custom algorithms
-----------------

To implement a custom algorithm, it is necessary to first understand how Composer uses `events` to
know where in the training loop to run an algorithm, and how algorithms can modify the `state` used for
subsequent computations.

.. `Events` denote locations inside the training procedure where algorithms can be run. In pseudocode,
.. Composer’s `events` look as follows:

.. ```python
.. EVENT.INIT
.. state.model = model()
.. state.train_dataloader = train_dataloader()
.. state.optimizers = optimizers()
.. EVENT.FIT_START
.. for epoch in epochs:
.. 	EVENT.EPOCH_START
.. 	for batch in state,train_dataloader:
.. 		EVENT.AFTER_DATALOADER
.. 		EVENT.BATCH_START
.. 		prepare_batch_for_training()
.. 		EVENT.BEFORE_TRAIN_BATCH

.. 		EVENT.BEFORE_FORWARD
.. 		forward_pass()
.. 		EVENT.AFTER_FORWARD

.. 		EVENT.BEFORE_LOSS
.. 		compute_loss()
.. 		EVENT.AFTER_LOSS

.. 		EVENT.BEFORE_BACKWARD
.. 		backward_pass()
.. 		EVENT.AFTER_BACKWARD

.. 		EVENT.AFTER_TRAIN_BATCH
.. 		optimizers.step()
.. 		EVENT.BATCH_END
.. 	EVENT.EPOCH_END
.. ```

.. Complete definitions of these events can be found [here](https://github.com/mosaicml/composer/blob/dev/composer/core/event.py). Some events have a `before` and `after` flavor. These events differ in the order that algorithms are run. For example, on `EVENT.BEFORE_X`, algorithms passed to the trainer in order `[A, B, C]` are also run in order `[A, B,C]`. On `EVENT.AFTER_X`, algorithms passed to the trainer in order `[A, B, C]` are run in order `[C, B, A]` . This allows algorithms to clean undo their effects on state if necessary.

.. Composer’s `state` tracks relevant quantities for the training procedure. The code for `state` can be found [here](https://github.com/mosaicml/composer/blob/dev/composer/core/state.py).  Algorithms can modify state, and therefore modify the training procedure.

.. To implement a custom algorithm, one needs to create a class that inherits from Composer’s `Algorithm` class, and implements a `match` methods that specifies which event(s) the algorithm should run on, and an `apply` function that specifies how the custom algorithm should modify quantities in `state`.

.. The `match` method simply takes `state` and the current `event` as an argument, determines whether or not the algorithm should run, and returns true if it should, false otherwise. In code, a simple  `match` might look like this:

.. ```python
.. def match(self, event, state):
..   return event in [Event.AFTER_DATALOADER, Event.AFTER_FORWARD]
.. ```

.. This will cause the algorithm to run on the `AFTER_DATALOADER` and `AFTER_FORWARD` events. Note that a given algorithm might run on multiple events.

.. The `apply` method also takes `state` and the current `event` as arguments. Based on this information, `apply` carries out the appropriate algorithm logic, and modifies `state` with the changes necessary. In code, an `apply` might look like this:

.. ```python
..   def apply(self, event, state, logger):
.. 		if event == Event.AFTER_DATALOADER:
.. 			state.batch = process_inputs(state.batch)
.. 		if event == Event.AFTER_FORWARD:
.. 			state.output = process_outputs(state.outputs)
.. ```

.. Note that different logic can be used for different events.

.. Packaging this all together into a class gives the object that Composer can run:

.. ```python
.. from composer.core import Algoritm, Event

.. class MyAlgorithm(Algorithm):
..   def __init__(self, hparam1=1):
..     self.hparam1 = hparam1

.. 	def match(self, event, state):
.. 	  return event in [Event.AFTER_DATALOADER, Event.AFTER_FORWARD]

..   def apply(self, event, state, logger):
.. 		if event == Event.AFTER_DATALOADER:
.. 			state.batch = process_inputs(state.batch, self.hparam1)
.. 		if event == Event.AFTER_FORWARD:
.. 			state.output = process_outputs(state.outputs)
.. ```

.. Using this in training can be done the same way as with Composer’s native algorithms.

.. ```python
.. from composer import Trainer
.. from composer.algorithms.blurpool import BlurPool
.. from composer.algorithms.channels_last import ChannelsLast

.. channels_last = ChannelsLast()
.. blurpool = BlurPool(replace_convs=True,
.. 										replace_maxpools=True,
.. 										blur_first=True)
.. custom_algorithm = MyAlgorithm(hparam1=1)

.. trainer = Trainer(model=model,
..                   train_dataloader=train_dataloader,
..                   eval_dataloader=test_dataloader,
..                   max_duration='90ep',
..                   device='gpu',
..                   algorithms=[channels_last, blurpool, custom_algorithm],
..                   validate_every_n_epochs=-1,
..                   seed=42)
.. ```