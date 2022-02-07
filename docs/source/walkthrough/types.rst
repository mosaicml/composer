composer.types
==============

.. currentmodule:: composer.types

This is our reference for common types used throughout our library.

Tensor Types
------------

.. py:class:: Tensor

    Alias for :class:`torch.Tensor`

.. py:class:: Tensors

    Commonly used to represent e.g. a set of inputs, where it is unclear whether each input has its own tensor,
    or if all the inputs are concatenated in a single tensor.


    Type: :class:`torch.Tensor`, tuple of :class:`torch.Tensor`\s, or list of :class:`torch.Tensor`\s

Batch Types
-----------

A batch of data can be represented in several formats, depending on the application.

.. py:class:: Batch

    Union type covering the most common representations of batches.

    Type: :class:`BatchPair`, :class:`BatchDict`, or :class:`torch.Tensor`

.. py:class:: BatchPair

    Commonly used in computer vision tasks. The object is assumed to contain exactly two elements,
    where the first represents inputs and the second represents targets.

    Type: List of :class:`Tensors`

.. py:class:: BatchDict

    Commonly used in natural language processing tasks.

    Type: str to :class:`Tensor` dict


Dataset and Data Loader Types
-----------------------------

.. py:class:: Dataset
    
    Alias for :class:`torch.utils.data.Dataset[Batch]`

.. autoclass:: DataLoader
    :members:

Trainer Types
-------------

.. py:class:: Metrics

    Union type covering common formats for representing metrics.

    Type: :class:`~torchmetrics.metric.Metric` or :class:`~torchmetrics.collections.MetricCollection`


.. py:class:: Optimizer

    Alias for :class:`torch.optim.Optimizer`

.. py:class:: Optimizers

    Union type for indeterminate amounts of optimizers.

    Type: :class:`Optimizer` or tuple of :class:`Optimizer`

.. py:class:: Scheduler

    Alias for :class:`torch.optim.lr_scheduler._LRScheduler`

.. py:class:: Schedulers

    Union type for indeterminate amounts of schedulers.

    Type: :class:`Scheduler` or tuple of :class:`Scheduler`

.. py:class:: Scaler

    Alias for :class:`torch.cuda.amp.grad_scaler.GradScaler`

.. py:class:: Model

    Alias for :class:`torch.nn.Module`

Miscellaneous Types
-------------------

.. py:class:: ModelParameters

    Type alias for model parameters used to initialize optimizers.

    Type: List of :class:`torch.Tensor` or list of str to :class:`torch.Tensor` dicts

.. py:class:: JSON

    JSON data.

.. autoclass:: Precision
    :members:

.. py:class:: TDeviceTransformFn

    Type alias for prefetch functions used for initializing dataloaders.

    Type: A function that takes a :class:`Batch` and returns a :class:`Batch`

.. autoclass:: Serializable
    :members:

.. py:class:: StateDict

    Class for representing serialized data.

    Type: dict with str keys.

.. autoexception:: BreakEpochException
