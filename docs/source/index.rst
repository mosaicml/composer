.. Composer documentation master file

Composer (Beta)
===============

MosaicML ``Composer`` contains a library of ML training efficiency methods, and a modular approach to compose them
together to train deep neural networks. We aim to ease the transition from research to industry through reproducible
code and rigorous benchmarking. With Composer, speed-up or accuracy-boosting methods can be easily composed into
complete recipes.

The library features:

* Implementation of 20+ efficiency methods curated from the research community
* Standardized approach to implement and compose efficiency methods, extended from two-way callbacks
  (`Howard et al, 2020 <https://arxiv.org/abs/2002.04688>`_)
* Easy way to access our methods either directly for your trainer loops, or through the Composer trainer

.. note::
    MosaicML Composer is currently in **beta**, so the API is subject to change.


Motivation
~~~~~~~~~~

MosaicML exists to make ML training more efficient. We believe large scale ML should be available to everyone
not just large companies.

The ML community is overwhelmed by the plethora of new algorithms in the literature and open source.
It is often difficult to integrate new methods into existing code, due to reproducibility
(`Pineau et al, 2020 <https://arxiv.org/abs/2003.12206>`_) and complexity. In addition, methods should be characterized
according to their effect of time-to-train and interactions with systems.

For more details on our philosophy, see our `Methodology <https://www.mosaicml.com/blog/methodology>`_ and our
`founder's blog <https://www.mosaicml.com/blog/founders-blog>`_.

We hope to contribute to the amazing community around ML Systems and ML Training efficiency.

Documentation
~~~~~~~~~~~~~

Our documentation is organized into a few sections:

* :doc:`Getting Started</getting_started/installation>` covers installation, a quick tour and
  explains how to use ``Composer``.
* :doc:`API Reference</api_reference>` contains the library's API reference.
* :doc:`Methods Library</method_cards/alibi>` details our implemented efficiency methods.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation.rst
   getting_started/using_composer.rst
   getting_started/welcome_tour.rst
   tutorials/adding_models_datasets.rst

.. toctree::
   :maxdepth: 1
   :caption: Using our Trainer

   trainer/summary.rst
   trainer/composer_model.rst
   trainer/algorithms.rst
   trainer/detailed_usage.rst


.. toctree::
   :maxdepth: 1
   :caption: Notebook Examples

   notebooks/summary.rst

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
   :maxdepth: 1
   :caption: Hparams System (YAHP)

   yahp/getting_started.rst

.. toctree::
   :caption: API Reference
   :maxdepth: 4

   api_reference.rst

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
