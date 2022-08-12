.. Composer documentation master file

Composer
========

`Composer <https://github.com/mosaicml/composer>`_ is a library for training neural networks better, faster, and cheaper.
It contains many state-of-the-art methods for accelerating neural network training and improving generalization, along with
an optional :doc:`Trainer<trainer/using_the_trainer>` API that makes `composing` many different enhancements easy.

We implemented dozens of papers and `trained thousands of models <https://app.mosaicml.com/explorer/imagenet>`_ so you don't have to,
and built strong baselines to rigorously measure their effect.

With Composer, you can build recipes to:

.. include:: ../../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- start numbers -->
   :end-before: <!-- end numbers -->


Composer features:

- 20+ curated efficiency methods for training better language and vision models.
- An easy-to-use Trainer interface written to be as performant as possible, with integrated multi-GPU training, mixed-precision training, and numerous `best practices <https://www.mosaicml.com/blog/5-best-practices-for-efficient-model-training>`_ implemented out of the box.
- Standalone functional forms of training methods for easy integration into your own training loop
- Strong, reproducible baselines to get you started as fast as possible.

See :doc:`getting_started/installation` for
how to install and start using Composer, :doc:`trainer/using_the_trainer` for an introduction
to our trainer, and :doc:`trainer/algorithms` for details about our efficiency methods and how to use them in your code.

At `MosaicML <https://mosaicml.com>`_, we are focused on making fast, high-quality ML training accessible to everyone. To do this,
we continually productionize state-of-the-art academic research on efficient model
training, and also study the `combinations` of these methods in order to ensure
that model training is ✨ as efficient as possible ✨.

If you have any questions, please feel free to reach out to us on `Twitter`_, `Email`_, or `Slack`_!

Composer is part of the broader Machine Learning community, and we welcome any contributions, pull requests, and issues.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation.rst
   getting_started/quick_start.rst
   getting_started/welcome_tour.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorials


   examples/getting_started.ipynb
   examples/functional_api.ipynb
   examples/medical_image_segmentation.ipynb
   examples/custom_speedup_methods.ipynb
   examples/nlp_models.ipynb
   examples/ffcv_dataloaders.ipynb
   examples/huggingface_models.ipynb
   examples/migrate_from_ptl.ipynb
   examples/early_stopping.ipynb
   examples/streaming_dataloader_facesynthetics.ipynb
   examples/auto_grad_accum.ipynb
   examples/checkpoint_autoresume.ipynb
   examples/training_without_local_storage.ipynb
   examples/exporting_for_inference.ipynb
   examples/glue/glue_entrypoint.ipynb
   examples/TPU_Training_in_composer.ipynb

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Speedup Methods

   trainer/algorithms.rst
   functional_api.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Trainer

   trainer/using_the_trainer.rst
   composer_model.rst
   trainer/dataloaders.rst
   trainer/evaluation.rst
   trainer/schedulers.rst
   trainer/time.rst
   trainer/events.rst
   trainer/checkpointing.rst
   trainer/logging.rst
   trainer/artifact_logging.rst
   trainer/callbacks.rst
   trainer/performance.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Notes

   notes/distributed_training.rst
   notes/early_stopping.rst
   notes/numerics.md
   notes/auto_grad_accum.rst
   notes/resumption.rst
   notes/tensorboard_logger.md
   notes/run_name.md

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Methods Library

   method_cards/methods_overview.rst
   method_cards/alibi.md
   method_cards/augmix.md
   method_cards/blurpool.md
   method_cards/channels_last.md
   method_cards/colout.md
   method_cards/cutmix.md
   method_cards/cutout.md
   method_cards/ema.md
   method_cards/factorize.md
   method_cards/fused_layernorm.md
   method_cards/gated_linear_units.md
   method_cards/ghost_batchnorm.md
   method_cards/gradient_clipping.md
   method_cards/label_smoothing.md
   method_cards/layer_freezing.md
   method_cards/mixup.md
   method_cards/progressive_resizing.md
   method_cards/randaugment.md
   method_cards/sam.md
   method_cards/scale_schedule.md
   method_cards/selective_backprop.md
   method_cards/seq_length_warmup.md
   method_cards/squeeze_excite.md
   method_cards/stochastic_depth.md
   method_cards/stochastic_depth_samplewise.md
   method_cards/swa.md

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Model Library

   model_cards/BERT.md
   model_cards/cifar_resnet.md
   model_cards/deeplabv3.md
   model_cards/efficientnet.md
   model_cards/GPT2.md
   model_cards/resnet.md
   model_cards/unet.md

.. toctree::
   :hidden:
   :caption: API Reference
   :maxdepth: 1

   api_reference.rst

.. _Twitter: https://twitter.com/mosaicml
.. _Email: mailto:community@mosaicml.com
.. _Slack: https://join.slack.com/t/mosaicml-community/shared_invite/zt-w0tiddn9-WGTlRpfjcO9J5jyrMub1dg
