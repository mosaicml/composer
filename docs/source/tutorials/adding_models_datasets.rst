Custom Models/datasets
======================

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

In this tutorial, we start with a simple image classification model from ``torchvision``:

.. code-block:: python

    import torchvision
    import composer

    class SimpleModel(composer.models.MosaicClassifier):
        def __init__(self):
            module = torchvision.models.resnet18()
            super().__init__(module=module)

Datasets
--------

Provide the trainer with your `torch.utils.data.Dataset` by configuring a dataloader spec for both train and eval datasets. Here, we create the :class:`~composer.datasets.DataloaderSpec` with the ``MNIST`` dataset:

.. code-block:: python

     from composer import models, datasets
     from torchvision import datasets

     train_dataloader_spec = datasets.DataloaderSpec(
         dataset=datasets.MNIST('/datasets/', train=True, download=True),
         drop_last=False,
         shuffle=True,
     )

     eval_dataloader_spec = datasets.DataloaderSpec(
         dataset=datasets.MNIST('/datasets/', train=False, download=True),
         drop_last=False,
         shuffle=False,
     )

Trainer init
------------

Now that your ``Dataset`` and ``Model`` are ready, you can initialize the :class:`~composer.trainer.Trainer` and train your model with our algorithms.

.. code-block:: python

     from composer import Trainer
     from composer.algorithms import LabelSmoothing, CutOut

     trainer = Trainer(
           model=SimpleModel()
           train_dataloader_spec=train_dataloader_spec,
           eval_dataloader_spec=eval_dataloader_spec,
           max_epochs=3,
           train_batch_size=256,
           eval_batch_size=256,
           algorithms=[
               CutOut(n_holes=1, length=10),
               LabelSmoothing(alpha=0.1).
            ]
       )

       trainer.fit()

Trainer with YAHP
-----------------





