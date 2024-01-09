.. Composer documentation master file

Composer
========

`Composer <https://github.com/mosaicml/composer>`_  is an open-source deep learning training
library by [MosaicML](https://www.mosaicml.com/). Built on top of PyTorch, the Composer library
makes it easier to implement distributed training workflows on large-scale clusters.

We built Composer to be **optimized for scalability and usability**, integrating best practices
for efficient, multi-node training. By abstracting away low-level complexities like parallelism
techniques, distributed data loading, and memory optimization, you can focus on training modern
ML models and running experiments without slowing down.

We recommend using Composer to speedup your experimentation workflow if you're training neural
networks of any size, including:

- Large Language Models (LLMs)
- Diffusion models
- Embedding models (e.g. BERT)
- Transformer-based models
- Convolutional Neural Networks (CNNs)

Composer is heavily used by the MosaicML research team to train state-of-the-art models like MPT,
and we open-sourced this library to enable the ML community to do the same. This framework is used
by organizations in both the tech industry and the academic sphere and is continually updated with
new features, bug fixes, and stability improvements for production workloads.

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
   examples/finetune_huggingface.ipynb
   examples/pretrain_finetune_huggingface.ipynb
   examples/migrate_from_ptl.ipynb
   examples/early_stopping.ipynb
   examples/auto_microbatching.ipynb
   examples/checkpoint_autoresume.ipynb
   examples/exporting_for_inference.ipynb
   examples/TPU_Training_in_composer.ipynb
   examples/training_with_submitit.ipynb
   tutorials/train_resnet50_on_aws.md

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
   trainer/file_uploading.rst
   trainer/callbacks.rst
   trainer/performance.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Notes

   notes/distributed_training.rst
   notes/early_stopping.rst
   notes/numerics.md
   notes/auto_microbatching.rst
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
   method_cards/decoupled_weight_decay.md
   method_cards/ema.md
   method_cards/factorize.md
   method_cards/gated_linear_units.md
   method_cards/ghost_batchnorm.md
   method_cards/gradient_clipping.md
   method_cards/gyro_dropout.md
   method_cards/label_smoothing.md
   method_cards/layer_freezing.md
   method_cards/low_precision_groupnorm.md
   method_cards/low_precision_layernorm.md
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
   method_cards/weight_standardization.md

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
   :glob:

   api_reference/*


.. _Twitter: https://twitter.com/mosaicml
.. _Email: mailto:community@mosaicml.com
.. _Slack: https://mosaicml.me/slack
