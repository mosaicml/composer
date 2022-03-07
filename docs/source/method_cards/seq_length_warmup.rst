|:high_brightness:| Sequence Length Warmup
==========================================

.. figure:: https://storage.googleapis.com/docs.mosaicml.com/images/methods/seq_len_warmup.svg
   :align: center
   :alt: alternate text
   :figclass: align-center

   An example plot showing applying sequence length warmup to 30% of the training duration.

Tags: ``Method``, ``Autoregressive Language Modeling``, ``Masked Language Modeling``, ``NLP``, ``Warmup``, ``Curriculum``, ``Speedup``, ``Decreased Wall Clock Time``


TL;DR
-----

Sequence Length Warmup warms up the sequence length (number of tokens)
from a ``min_seq_length`` to a ``max_seq_length`` over some duration of
training. The underlying motivation is that sequence length is a proxy
for the difficulty of an example. Sequence Length Warmup is able to reduce
training time by ~1.5x while still achieving the same loss as baseline
models.

Hyperparameters
---------------

-  ``duration`` - The fraction of training that the warmup should be
   applied for.
-  ``min_seq_length`` - The initial sequence length.
-  ``max_seq_length`` - The final sequence length. Used for the rest of
   training.
-  ``step_size`` - The number of tokens to increase the sequence length
   by at each step. Multiples of 8 are preferred in order to enable
   hardware acceleration.
-  ``truncate`` - How the sequence length adjustment is achieved.
   ``False`` reshapes the data tensor, creating new samples out of the
   extra tokens. ``True`` truncates the tensor, discarding the extra
   tokens.

Applicable Settings
-------------------

Sequence Length Warmup as implemented in Composer applies to language
modeling tasks, including autoregressive language modeling and masked
language modeling.


Effects
---------------

Our experiments found that Sequence Length Warmup could speed up
training by a factor of ~1.5x while achieving the same loss. The
original authors of the paper claim that Sequence Length Warmup reduces
the outliers in Adam's (`Kingma and Ba <https://arxiv.org/abs/1412.6980>`__)
variance term, which permits training on larger batch sizes and larger
learning rates without divergence.

Implementation Details
----------------------

Warmup Implementation
~~~~~~~~~~~~~~~~~~~~~

We implement this as a processing step during the forward pass, where we
can either:

1. Truncate the tensor at the sequence length specified by
   the warmup schedule.
2. Reshape the tensor to the sequence length specified by the warmup,
   which allocates the extra tokens along the batch dimension.

**Example when** ``truncate = True`` **and** ``seq_len = 8`` **:**

*Original Input (2 samples):*

.. code-block:: none

    We choose to go to the moon. We choose to go to the moon in this decade and do the other things, not because they are easy, but because they are hard, because that goal will serve to organize and measure the best of our energies and skills.

    It is for these reasons that I regard the decision last year to shift our efforts in space from low to high gear as among the most important decisions that will be made during my incumbency in the office of the Presidency.

*Transformed Inputs (2 samples):*

.. code-block::

    We choose to go to the moon.

    It is for these reasons that I regard

**Example when** ``truncate = False`` **and** ``seq_len = 8`` **:**

*Original Input (2 samples):*

.. code-block::

    We choose to go to the moon. We choose to go to the moon in this decade and do the other things, not because they are easy, but because they are hard, because that goal will serve to organize and measure the best of our energies and skills, because that challenge
    It is for these reasons that I regard the decision last year to shift our efforts in space from low to high gear as among the most important decisions that will be made during my incumbency in the office of the Presidency.

*Transformed Inputs (14 samples):*

.. code-block::

    We choose to go to the moon.
    We choose to go to the moon in
    this decade and do the other things
    not because they are easy, but because
    they are hard, because that goal will
    serve to organize and measure the best of
    our energies and skills, because that challenge

    It is for these reasons that I regard
    the decision last year to shift our efforts
    in space from low to high gear as
    among the most important decisions that will be
    made during my incumbency in the office
    of the Presidency. In the last 24
    hours we have seen facilities now being created

Avoiding Out-Of-Memory Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sequence Length Warmup starts with a small sequence length and gradually
increases it. However, as a result, it constantly requires PyTorch to
expand its memory cache allocation with new buffers as larger tensor
sizes are consistently being streamed in.

In order to address this we create dummy inputs to the model, perform a
forward and backward pass, and zero out the gradients. We do
this *without taking any scheduler or optimization steps*. This permits
PyTorch to allocate buffers for the maximum possible sequence length,
and help avoid downstream out-of-memory errors.

Suggested Hyperparameters
-------------------------

We swept the ``duration`` from ``0.0`` to ``0.9`` in increments of
``0.1`` across the ``GPT-2 52M`` model, and found that running the
sequence length warmup for 30% of training leads to the fastest wall clock time to
reach the same loss. This corroborates the suggested hyperparameters in
the paper,

Considerations
--------------

Sequence length warmup is a form of curriculum learning, a category of
techniques that present samples in a structured or organized order, such
as by difficulty. Accordingly, it may compose poorly with other
curriculum learning techniques such as batch-size warmup, which is used
in the `GPT-3 paper <https://arxiv.org/abs/2005.14165>`__.

Composition
-----------

This method composes well with ALiBi (Press et al., 2021), a method that
enables good extrapolation from shorter training sequence lengths to
longer evaluation sequence lengths.

Attribution
-----------

`Curriculum Learning: A Regularization Method for Efficient and Stable
Billion-Scale GPT Model Pre-Training <https://arxiv.org/abs/2108.06084>`__ by Conglong Li,
Minjia Zhang, and Yuxiong He. Posted to arXiv in 2021.


Code
----

.. autoclass:: composer.algorithms.seq_length_warmup.SeqLengthWarmup
    :members: match, apply

.. autoclass:: composer.algorithms.seq_length_warmup.set_batch_sequence_length
