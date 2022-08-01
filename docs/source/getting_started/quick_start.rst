|:rocket:| Quick Start
======================

Access our library of speedup methods with the :doc:`/functional_api` API:

.. testcode::

    import logging
    from composer import functional as CF
    import torchvision.models as models

    logging.basicConfig(level=logging.INFO)
    model = models.resnet50()

    CF.apply_blurpool(model)

This creates a ResNet50 model and replaces several pooling and convolution layers with
BlurPool variants (`Zhang et al, 2019 <https://arxiv.org/abs/1904.11486>`_). For more information,
see :doc:`/method_cards/blurpool`. The method should log:

.. code-block:: none

    Applied BlurPool to model ResNet. Model now has 1 BlurMaxPool2d and 6 BlurConv2D layers.

These methods are easy to integrate into your own training loop code with just a few lines.

For an overview of the algorithms, see :doc:`/trainer/algorithms`.

We make composing recipes together even easier with our (optional) :class`.Trainer`. Here,
we train an MNIST classifer with a recipe of methods:

.. code-block:: python

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    from composer import Trainer
    from composer.models import mnist_model
    from composer.algorithms import LabelSmoothing, CutMix, ChannelsLast

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=128)

    trainer = Trainer(
        model=mnist_model(num_classes=10),
        train_dataloader=train_dataloader,
        max_duration="2ep",
        algorithms=[
            LabelSmoothing(smoothing=0.1),
            CutMix(alpha=1.0),
            ChannelsLast(),
            ]
    )
    trainer.fit()

We handle inserting and running the logic during the training so that any algorithms you specify "just work."

Besides easily running our built-in algorithms, Composer also features:

* An interface to flexibly add algorithms to the training loop
* An engine that manages the ordering of algorithms for composition
* A trainer to handle boilerplate around numerics, distributed training, and others
* Integration with popular model libraries such as TIMM and HuggingFace Transformers

Next steps
----------

* Try our :doc:`/examples/getting_started` tutorial on Colab.
* See :doc:`/trainer/using_the_trainer` for more details on our trainer.
* Read :doc:`/getting_started/welcome_tour` for a tour through the library.
