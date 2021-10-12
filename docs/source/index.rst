.. mosaicml documentation master file

Composer (Beta)
===============

MosaicML ``Composer`` contains a library of ML training efficiency methods, and a modular approach to compose them together to train deep neural networks. We aim to ease the transition from research to industry through reproducible code and rigorous benchmarking. With Composer, speed-up or accuracy-boosting methods can be easily composed into complete recipes.

The library features:

* Implementation of 20+ efficiency methods curated from the research community
* Standardized approach to implement and compose efficiency methods, extended from two-way callbacks (`Howard et al, 2020 <https://arxiv.org/abs/2002.04688>`_)
* Easy way to access our methods either directly for your trainer loops, or through the MosaicML trainer

.. note::
    MosaicML Composer is currently in **beta**, so the API is subject to change.

MosaicML has been tested with Ubuntu 20.04, PyTorch 1.8.1, and Python 3.8.


Motivation
~~~~~~~~~~

MosaicML exists to make ML training more efficient. We believe large scale ML should be available to everyone not just large companies.

The ML community is overwhelmed by the plethora of new algorithms in the literature and open source. It is often difficult to integrate new methods into existing code, due to reproducibility (`Pineau et al, 2020 <https://arxiv.org/abs/2003.12206>`_) and complexity. In addition, methods should be charaterized according to their effect of time-to-train and interactions with systems.

For more details on our philosophy, see <> and <>.

We hope to contribute to the amazing community around ML Systems and ML Training efficiency.

Documentation
~~~~~~~~~~~~~

Our documentation is organized into a few sections:

* :doc:`Getting Started </getting_started/installation>` covers installation, a quick tour and explains the ways to use Composer
* :doc:`Core </core/algorithm>` covers the core API of our algorithms.
* :doc:`composer <hparams_reference>` contains the library API
* :doc:`Science </science/methodology>` is a primer on our research methodology


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation.rst
   getting_started/using_composer.rst
   getting_started/quick_tour.rst
   tutorials/adding_models_datasets.rst
   tutorials/adding_algorithms.rst
   tutorials/adding_loggers_callbacks.rst
   tutorials/yahp.rst

.. toctree::
   :maxdepth: 1
   :caption: core

   core/functional.rst
   core/state.rst
   core/algorithm.rst
   core/callback.rst
   core/logging.rst
   core/event.rst
   core/engine.rst
   core/surgery.rst

.. toctree::
   :maxdepth: 1
   :caption: composer

   hparams_reference.rst
   algorithms.rst
   callbacks.rst
   core/surgery.rst
   core/types.rst
   core/misc.rst
   datasets.rst
   functional.rst
   loggers.rst
   models.rst
   optim.rst
   trainer.rst
   trainer_utils.rst
   trainer_devices.rst
   utils.rst

.. toctree::
   :maxdepth: 1
   :caption: Science

   science/methodology.rst
   science/glossary.rst

.. toctree::
   :maxdepth: 1
   :caption: Methods Library

   method_cards/methods_intro.rst
   method_cards/aug_mix.md
   method_cards/block_wise_stochastic_depth
   method_cards/label_smoothing.md
   method_cards/mix_up.md
   method_cards/progressive_resizing_vision.md
   method_cards/rand_augment.md

.. toctree::
   :maxdepth: 1
   :caption: Model Library

   model_cards/model_intro.rst

.. toctree::
   :maxdepth: 1
   :caption: Datasets Library

   dataset_cards/dataset_intro.rst


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
