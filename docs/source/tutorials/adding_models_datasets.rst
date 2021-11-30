Custom Models and Datasets
==========================

The MosaicML :class:`~composer.trainer.Trainer` can easily be extended to use your own models and datasets. We walk through two ways to get started and experiment with algorithms on your machine learning projects.

.. seealso::

    The :doc:`../functional` API can be used to directly call the efficiency methods into your trainer loop. The :doc:`../trainer` described imposes a minimal level of overhead to enable access to composability and configuration management.

Models
------

Models provided to :class:`~composer.trainer.Trainer` use the minimal interface in :class:`~composer.models.BaseMosaicModel`:

.. code-block:: python

    class BaseMosaicModel(torch.nn.Module, ABC):

        def forward(self, batch: Batch) -> Tensors:
        # computes the forward pass given a batch of data.

        def loss(self, outputs: Any, batch: Batch) -> Tensors:
        # given the outputs from forward, and the batch, return the loss

        def metrics(self, train: bool = False) -> Metrics:
        # returns a collection of `torchmetrics`

        def validate(self, batch: Batch) -> Tuple[Any, Any]:
        # runs validation and returns a tuple of results that are
        # then passed to self.metrics

.. note::

    The ``Batch`` is the data returned from your ``dataloader``. Since our algorithms need to know the structure of ``Batch`` in order to apply itself (e.g. augmentations must access the inputs), we currently support two types of ``Batch``: a tuple of ``(input, target)`` tensors, and a `Dict[str, Tensor]` typically used for NLP applications.

For convenience, we've provided a few base classes that are task-specific:

* Classification: :class:`~composer.models.MosaicClassifier`. Uses cross entropy loss and `torchmetrics.Accuracy`.
* Transformers: :class:`~composer.models.MosaicTransformer`. For use with HuggingFace Transformers.
* Segmentation: :class:`~composer.models.unet.UNet`. Uses a Dice and CE loss.

In this tutorial, we start with a simple image classification model:

.. code-block:: python

    import torch
    import composer

    class SimpleModel(composer.models.MosaicClassifier):
        def __init__(self, num_hidden: int, num_classes: int):
            module = torch.nn.Sequential(
                torch.nn.Flatten(start_dim=1),
                torch.nn.Linear(28 * 28, num_hidden),
                torch.nn.Linear(num_hidden, num_classes),
            )
            self.num_classes = num_classes
            super().__init__(module=module)

Datasets
--------

Provide the trainer with :class:`~torch.utils.data.DataLoader` for both 
train and validation datasets. Here, we create a :class:`~torch.utils.data.DataLoader` with the ``MNIST`` dataset:

.. code-block:: python

     from torchvision import datasets, transforms
     from torch.utils.data import DataLoader

     train_dataloader = DataLoader(
         dataset=datasets.MNIST('/datasets/', train=True, transform=transforms.ToTensor(), download=True),
         drop_last=False,
         shuffle=True,
         batch_size=256,
     )

     eval_dataloader = DataLoader(
         dataset=datasets.MNIST('/datasets/', train=False, transform=transforms.ToTensor()),
         drop_last=False,
         shuffle=False,
         batch_size=256,
     )

Trainer init
------------

Now that your ``Dataset`` and ``Model`` are ready, you can initialize the :class:`~composer.trainer.Trainer` and train your model with our algorithms.

.. code-block:: python

    from composer import Trainer
    from composer.algorithms import LabelSmoothing, CutOut

    trainer = Trainer(
        model=SimpleModel(num_hidden=128, num_classes=10),
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_epochs=3,
        algorithms=[
            CutOut(n_holes=1, length=10),
            LabelSmoothing(alpha=0.1),
        ]
    )

    trainer.fit()

Trainer with YAHP
-----------------

Integrating your models and datasets with :mod:`yahp.hparams` allows for configuration via ``yaml`` or command line flags automagically. This is recommended if you are running experiments or large scale runs, to ensure reproducibility.

First, create :class:`~yahp.hparams.Hparams` dataclasses for both your model and your dataset:

.. code-block:: python

    from dataclasses import dataclass
    from composer import models, datasets
    import yahp as hp

    @dataclass
    class MyModelHparams(models.ModelHparams):

        num_hidden: int = hp.optional(doc="num hidden features", default=128)
        num_classes: int = hp.optional(doc="num of classes", default=10)

        def initialize_object(self):
            return SimpleModel(
                num_hidden=self.num_hidden,
                num_classes=self.num_classes
            )

    @dataclass
    class MNISTHparams(datasets.DatasetHparams):
        is_train: bool = hp.required("whether to load the training or validation dataset")
        datadir: str = hp.required("data directory")
        download: bool = hp.required("whether to download the dataset, if needed")
        drop_last: bool = hp.optional("Whether to drop the last samples for the last batch", default=True)
        shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch", default=True)

        def initialize_object(self) -> DataloaderSpec:
            transform = transforms.Compose([transforms.ToTensor()])
            dataset = datasets.MNIST(
                self.datadir,
                train=self.is_train,
                download=self.download,
                transform=transform,
            )
            return DataloaderSpec(
                dataset=dataset,
                drop_last=self.drop_last,
                shuffle=self.shuffle,
            )

Then, we can register them with the trainer:

.. code-block:: python

    from composer.trainer import TrainerHparams

    TrainerHparams.register_class(
        field='model',
        register_class=MyModelHparams,
        class_key='my_model'
    )

    dataset_args = {
       'register_class': MNISTHparams,
       'class_key': 'my_mnist'
    }
    TrainerHparams.register_class(
        field='train_dataset',
        **dataset_args
    )
    TrainerHparams.register_class(
        field='val_dataset',
        **dataset_args
    )

Now, your registered dataset and model is now available by invocation either in a ``yaml`` file:

.. code-block::

    model:
      my_model:
        num_classes: 10
        num_hidden: 128

or via the command line, e.g.

.. code-block::

    python examples/run_mosaic_trainer.py -f my_config.yaml --model my_model --num_classes 10 --num_hidden 128









