.. Composer documentation master file

Composer
========

Composer provides well-engineered implementations of efficient training methods
to give the tools that help you train a better model for cheaper.

Using Composer, you can:

- Train an ImageNet model to 76.1% accuracy for $37 (`with vanilla PyTorch`:$127)
- Train a GPT-2 125M to a perplexity of 23.9 for $148 (`with vanilla PyTorch`: $255)
- Use start-of-the-art implementations of methods to speed up your own training loop.

Composer features:

- 20+ efficient training methods for training a better language and vision models! Don't waste hours trying to reproduce research papers when Composer has done the work for you.
- Easy-to-use Trainer interface written to be as performant as possible, and `integrated best practices <https://www.mosaicml.com/blog/best-practices-dec-2021>`_.
- Easy-to-use Functional forms that allow you to integrate efficient training methods into your training loop!
- Strong, `reproducible` baselines to get you started as 💨 fast 💨 as possible

See :doc:`Getting Started<getting_started/installation>` for
installation an initial usage, the :doc:`Trainer<trainer/using_the_trainer>` section for an introduction
to our trainer, and :doc:`Methods<method_cards/methods_overview>` for details about our efficiency methods
and how to use them in your code.

At MosaicML, we are focused on making training ML models accessible. To do this,
we continually productionize state-of-the-art academic research on efficient model
training, and also study the `combinations`` of these methods in order to ensure
that model training is ✨ as efficient as possible ✨.

If you have any questions, please feel free to reach out to us on `Twitter`_, `Email`_, or join
our `Slack`_ channel!

Composer is part of the broader Machine Learning community, and we welcome any contributions, pull requests, or issues.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation.rst
   getting_started/using_composer.rst
   getting_started/welcome_tour.rst

.. toctree::
   :maxdepth: 1
   :caption: Trainer

   trainer/using_the_trainer.rst
   trainer/dataloaders.rst
   trainer/evaluation.rst
   trainer/schedulers.rst
   trainer/time.rst
   trainer/events.rst
   trainer/checkpointing.rst
   trainer/logging.rst
   trainer/callbacks.rst
   trainer/distributed_training.rst
   trainer/numerics.rst
   trainer/performance.rst

.. toctree::
   :maxdepth: 1
   :caption: Key Classes

   composer_model.rst
   trainer/algorithms.rst

.. toctree::
   :maxdepth: 1
   :caption: Methods Library

   method_cards/methods_overview.rst
   method_cards/alibi.md
   method_cards/augmix.md
   method_cards/blurpool.md
   method_cards/channels_last.md
   method_cards/cutmix.md
   method_cards/colout.md
   method_cards/cutout.md
   method_cards/decoupled_weight_decay.md
   method_cards/factorize.md
   method_cards/ghost_batchnorm.md
   method_cards/label_smoothing.md
   method_cards/layer_freezing.md
   method_cards/mixup.md
   method_cards/progressive_resizing.md
   method_cards/randaugment.md
   method_cards/scale_schedule.md
   method_cards/scaling_laws.rst
   method_cards/selective_backprop.md
   method_cards/seq_length_warmup.rst
   method_cards/sam.md
   method_cards/squeeze_excite.md
   method_cards/stochastic_depth.md
   method_cards/stochastic_depth_samplewise.md
   method_cards/swa.md

.. toctree::
   :maxdepth: 1
   :caption: Model Library

   model_cards/cifar_resnet.md
   model_cards/efficientnet.md
   model_cards/GPT2.md
   model_cards/resnet.md
   model_cards/unet.md

.. toctree::
   :caption: API Reference
   :maxdepth: 1

   api_reference.rst

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _Twitter: https://twitter.com/mosaicml
.. _Email: mailto:community@mosaicml.com
.. _Slack: https://join.slack.com/t/mosaicml-community/shared_invite/zt-w0tiddn9-WGTlRpfjcO9J5jyrMub1dg
