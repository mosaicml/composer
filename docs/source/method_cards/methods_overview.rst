|:black_joker:| Methods Overview
================================

.. _method_cards:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Method
     - Domains
     - Description
   * - `Alibi <alibi.html>`_
     - NLP
     - Replace attention with AliBi
   * - `AugMix <augmix.html>`_
     - CV
     - Image-preserving data augmentations
   * - `BlurPool <blurpool.html>`_
     - CV
     - Applies blur before pooling or downsampling
   * - `ChannelsLast <channels_last.html>`_
     - CV
     - Uses channels last memory format (NHWC)
   * - `ColOut <colout.html>`_
     - CV
     - Removes columns and rows from the image for augmentation and efficiency
   * - `CutMix <cutmix.html>`_
     - CV
     - Combines pairs of examples in non-overlapping regions and mixes labels
   * - `CutOut <cutout.html>`_
     - CV
     - Randomly erases rectangular blocks from the image
   * - `EMA <ema.html>`_
     - CV, NLP
     - Maintains an exponential moving average of model weights for use in evaluation
   * - `Factorize <factorize.html>`_
     - CV, NLP
     - Factorize GEMMs into smaller GEMMs
   * - `GatedLinearUnits <gated_linear_units.html>`_
     - NLP
     - Swaps the building block from a Linear layer to a Gated Linear layer
   * - `GhostBatchNorm <ghost_batchnorm.html>`_
     - CV
     - Use smaller # samples to compute batchnorm
   * - `GradientClipping <gradient_clipping.html>`_
     - CV, NLP
     - Clips all gradients in model based on specified clipping_type
   * - `GyroDropout <gyro_dropout.html>`_
     - CV
     - Clips all gradients in model based on specified clipping_type
   * - `LabelSmoothing <label_smoothing.html>`_
     - CV
     - Smooths the labels with a uniform prior
   * - `LayerFreezing <layer_freezing.html>`_
     - CV, NLP
     - Progressively freezes layers during training
   * - `LowPrecisionGroupNorm <low_precision_groupnorm.html>`_
     - CV, NLP
     - Forces GroupNorm to run in lower precision
   * - `LowPrecisionLayerNorm <low_precision_layernorm.html>`_
     - NLP, CV
     - Forces LayerNorm to run in lower precision
   * - `MixUp <mixup.html>`_
     - CV
     - Blends pairs of examples and labels
   * - `ProgressiveResizing <progressive_resizing.html>`_
     - CV
     - Increases the input image size during training
   * - `RandAugment <randaugment.html>`_
     - CV
     - Applies a series of random augmentations
   * - `SAM <sam.html>`_
     - CV
     - SAM optimizer measures sharpness of optimization space
   * - `SelectiveBackprop <selective_backprop.html>`_
     - CV
     - Drops examples with small loss contributions
   * - `SeqLengthWarmup <seq_length_warmup.html>`_
     - NLP
     - Progressively increase sequence length
   * - `SqueezeExcite <squeeze_excite.html>`_
     - CV
     - Replaces eligible layers with Squeeze-Excite layers
   * - `StochasticDepth <stochastic_depth.html>`_
     - CV
     - Replaces a specified layer with a stochastic version that randomly drops the layer or samples during training
   * - `SWA <swa.html>`_
     - CV, NLP
     - Computes running average of model weights
   * - `Weight Standardization <weight_standardization.html>`_
     - CV
     - Makes convolution weights always have zero mean and unit variance