|:robot:| Algorithms
====================

Composer has a curated collection of speedup methods ("Algorithms") that can be composed
to easily create efficient training recipes.

Below is a brief overview of the algorithms currently in Composer.
For more detailed information about each algorithm, see the :doc:`method cards</method_cards/methods_overview>`,
also linked in the table. Each algorithm has a functional implementation intended
for use with your own training loop and an implementation intended for use with
Composer's trainer.

.. csv-table::
    :header: "Name" , "tldr", "functional"
    :delim: |
    :widths: 30, 40, 30

    {% for name, data in metadata.items()|sort(attribute='1.name') %}
    {% if data.functional %}
    :doc:`{{ data.class_name }}</method_cards/{{name}}>` | {{ data.tldr }} | :func:`~composer.functional.{{ data.functional }}`
    {% else %}
    :doc:`{{ data.class_name }}</method_cards/{{name}}>` | {{ data.tldr }} | {{ data.functional }}
    {% endif %}
    {% endfor %}

Functional API
--------------

The simplest way to use Composer's algorithms is via the functional API.
Composer's algorithms can be grouped into three, broad classes:

- `data augmentations` add additional transforms to the training data.
- `model surgery` algorithms modify the network architecture.
- `training loop modifications` change various properties of the training loop.

Data Augmentations
~~~~~~~~~~~~~~~~~~

Data augmentations can be inserted into your `dataset.transforms` similiar to Torchvision's
transforms. For example, with :doc:`/method_cards/randaugment`:

.. code-block:: python

    import torch
    from torchvision import datasets, transforms

    from composer import functional as cf

    c10_transforms = transforms.Compose([cf.randaugment_image(), # <---- Add RandAugment
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    dataset = datasets.CIFAR10('../data',
                            train=True,
                            download=True,
                            transform=c10_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024)

Some augmentations, such as :doc:`/method_cards/cutmix`, act on a batch of inputs. Insert
these in your training loop after a batch is loaded from the dataloader:

.. code-block:: python

    from composer import functional as cf

    cutmix_alpha = 1
    num_classes = 10
    for batch_idx, (data, target) in enumerate(dataloader):
        data = cf.cutmix(
            data,
            target,
            alpha=cutmix_alpha,
            num_classes=num_classes
        )
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

Model Surgery
~~~~~~~~~~~~~

Model surgery algorithms make direct modifications to the network itself.
For example, apply :doc:`/method_cards/blurpool`, inserts a blur layer before strided convolution
layers as demonstrated here:

.. code-block:: python

    from composer import functional as cf
    import torchvision.models as models

    model = models.resnet18()
    cf.apply_blurpool(model)

For a transformer model, we can swap out the attention head of a |:hugging_face:| transformer with one
from :doc:`/method_cards/alibi`:

.. code-block:: python

    from composer import functional as cf
    from composer.algorithms.alibi.gpt2_alibi import _attn
    from composer.algorithms.alibi.gpt2_alibi import enlarge_mask

    from transformers import GPT2Model
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


    model = GPT2Model.from_pretrained("gpt2")

    cf.apply_alibi(
        model=model,
        heads_per_layer=12,
        max_sequence_length=8192,
        position_embedding_attribute="module.transformer.wpe",
        attention_module=GPT2Attention,
        attr_to_replace="_attn",
        alibi_attention=_attn,
        mask_replacement_function=enlarge_mask
    )


Training Loop
~~~~~~~~~~~~~

Methods such as :doc:`/method_cards/progressive_resizing` or :doc:`/method_cards/layer_freezing`
apply changes to the training loop. See their method cards for details on how to use them
in your own code.


Composer Trainer
----------------

Building training recipes require composing all these different methods together, which is
the purpose of our :class:`.Trainer`. Pass in a list of the algorithm classes to run
to the trainer, and we will automatically run each one at the appropriate time during training,
handling any collisions or reorderings as needed.

.. code-block:: python

    from composer import Trainer
    from composer.algorithms import BlurPool, ChannelsLast

    trainer = Trainer(
        model=model,
        algorithms=[ChannelsLast(), BlurPool()]
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        max_duration='10ep',
    )

For more information, see: :doc:`/trainer/using_the_trainer` and :doc:`/getting_started/welcome_tour`.


Two-way callbacks
-----------------

The way our algorithms insert themselves in our trainer is based on the two-way callbacks system developed
by (`Howard et al, 2020 <https://arxiv.org/abs/2002.04688>`__). Algorithms interact with the
training loop at various :class:`Events <.Event>` and effect their changes by modifing the trainer :class:`.State`.

`Events` denote locations inside the training procedure where algorithms can be run. In pseudocode,
Composer’s `events` look as follows:

.. code-block:: python

    EVENT.INIT
    state.model = model()
    state.train_dataloader = train_dataloader()
    state.optimizers = optimizers()
    load_checkpoint()
    EVENT.AFTER_LOAD
    EVENT.FIT_START
    for epoch in epochs:
        EVENT.EPOCH_START
        for batch in state.train_dataloader:
            EVENT.AFTER_DATALOADER
            EVENT.BATCH_START
            prepare_batch_for_training()
            EVENT.BEFORE_TRAIN_BATCH

            EVENT.BEFORE_FORWARD
            forward_pass()
            EVENT.AFTER_FORWARD

            EVENT.BEFORE_LOSS
            compute_loss()
            EVENT.AFTER_LOSS

            EVENT.BEFORE_BACKWARD
            backward_pass()
            EVENT.AFTER_BACKWARD

            EVENT.AFTER_TRAIN_BATCH
            optimizers.step()
            EVENT.BATCH_END
        EVENT.EPOCH_END


Complete definitions of these events can be found `here <https://github.com/mosaicml/composer/blob/dev/composer/core/event.py>`__. Some events have a `before` and `after` flavor. These events differ in the order that algorithms are run. For example, on `EVENT.BEFORE_X`, algorithms passed to the trainer in order `[A, B, C]` are also run in order `[A, B,C]`. On `EVENT.AFTER_X`, algorithms passed to the trainer in order `[A, B, C]` are run in order `[C, B, A]` . This allows algorithms to clean undo their effects on state if necessary.

Composer’s `state` tracks relevant quantities for the training procedure. The code for `state` can be found `here <https://github.com/mosaicml/composer/blob/dev/composer/core/state.py>`__. Algorithms can modify state, and therefore modify the training procedure.

To implement a custom algorithm, one needs to create a class that inherits from Composer’s `Algorithm` class, and implements a `match` methods that specifies which event(s) the algorithm should run on, and an `apply` function that specifies how the custom algorithm should modify quantities in `state`.

The `match` method simply takes `state` and the current `event` as an argument, determines whether or not the algorithm should run, and returns true if it should, false otherwise. In code, a simple  `match` might look like this:

.. code-block:: python

    def match(self, event, state):
    return event in [Event.AFTER_DATALOADER, Event.AFTER_FORWARD]

This will cause the algorithm to run on the `AFTER_DATALOADER` and `AFTER_FORWARD` events. Note that a given algorithm might run on multiple events.

The `apply` method also takes `state` and the current `event` as arguments. Based on this information, `apply` carries out the appropriate algorithm logic, and modifies `state` with the changes necessary. In code, an `apply` might look like this:

.. code-block:: python

    def apply(self, event, state, logger):
        if event == Event.AFTER_DATALOADER:
            state.batch = process_inputs(state.batch)
        if event == Event.AFTER_FORWARD:
            state.output = process_outputs(state.outputs)


Note that different logic can be used for different events.

Packaging this all together into a class gives the object that Composer can run:

.. code-block:: python

    from composer.core import Algoritm, Event

    class MyAlgorithm(Algorithm):
    def __init__(self, hparam1=1):
        self.hparam1 = hparam1

        def match(self, event, state):
        return event in [Event.AFTER_DATALOADER, Event.AFTER_FORWARD]

    def apply(self, event, state, logger):
            if event == Event.AFTER_DATALOADER:
                state.batch = process_inputs(state.batch, self.hparam1)
            if event == Event.AFTER_FORWARD:
                state.output = process_outputs(state.outputs)


Using this in training can be done the same way as with Composer’s native algorithms.

.. code-block:: python

    from composer import Trainer
    from composer.algorithms.blurpool import BlurPool
    from composer.algorithms.channels_last import ChannelsLast

    channels_last = ChannelsLast()
    blurpool = BlurPool(replace_convs=True, replace_maxpools=True, blur_first=True)
    custom_algorithm = MyAlgorithm(hparam1=1)

    trainer = Trainer(model=model,
                    train_dataloader=train_dataloader,
                    eval_dataloader=test_dataloader,
                    max_duration='90ep',
                    device='gpu',
                    algorithms=[channels_last, blurpool, custom_algorithm],
                    eval_interval="0ep",
                    seed=42)
