|:art:| Using Composer
======================

Composer provides both a **Functional API** (similar to :mod:`torch.nn.functional`) and a
**Trainer** (that abstracts away the training loop) to provide flexibility to users.


Functional API
~~~~~~~~~~~~~~

For users who choose to use their own training loop, we provide state-less functional
implementations of our algorithms for a end-user to integrate.

The following example highlights using [BlurPool](https://arxiv.org/abs/1904.11486),
which applies an anti-aliasing filter before every downsampling operation.

.. code-block:: python

    from composer import functional as cf
    import torchvision

    model = torchvision.models.resnet50()

    # Apply model surgery before training by replacing eligible layers
    # with a BlurPool-enabled layer (Zhang, 2019)
    model = cf.apply_blurpool(model)

    # Start your training loop here
    train_loop(model)


As another example, to apply Progressive Resizing, which increases the
image size over the course of training:

.. code-block:: python

    from composer import functional as CF

    scale = 0.5
    for (image, label) in your_dataloader:
        CF.resize_inputs(image, label, scale_factor=scale)
        scale += 0.01

        # your train step here
        train_step()

For more details, please see :mod:`composer.functional`.

.. _using_composer_trainer:

Composer Trainer
~~~~~~~~~~~~~~~~

For maximal speedups, we recommend using our Trainer, which manages handling user state,
performant algorithm implementations, and provides useful engineering abstractions to permit
rapid experimentation.

Our trainer features:

* interface to flexibly add algorithms to the training loop
* engine that manages the ordering of algorithms for composition
* trainer to handle boilerplate around numerics, distributed training, and others
* integration with popular model libraries such as TIMM or HuggingFace Transformers.

For more details, see :doc:`Using Trainer</trainer/using_the_trainer>`


.. _yahp: https://github.com/mosaicml/yahp
