
|:black_joker:| Methods Overview
================================

.. panels::
   :card: shadow

   ---
   .. link-button:: agc.html
      :text: AGC
      :classes: card-title

   :badge:`CV,badge-primary`
   ++++++++++++++
   Clips gradients based on the ratio of their norms with weights' norms.

   ---
   .. link-button:: alibi.html
      :text: Alibi
      :classes: card-title

   :badge:`NLP,badge-success`
   ++++++++++++++
   Replace attention with AliBi

   ---
   .. link-button:: augmix.html
      :text: AugMix
      :classes: card-title

   :badge:`CV,badge-primary`
   ++++++++++++++
   Image-preserving data augmentations

   ---
   .. link-button:: blurpool.html
      :text: BlurPool
      :classes: card-title

   :badge:`CV,badge-primary`
   ++++++++++++++
   Applies blur before pooling or downsampling

   ---
   .. link-button:: channels_last.html
      :text: ChannelsLast
      :classes: card-title

   :badge:`CV,badge-primary`
   ++++++++++++++
   Uses channels last memory format (NHWC)

   ---
   .. link-button:: colout.html
      :text: ColOut
      :classes: card-title

   :badge:`CV,badge-primary`
   ++++++++++++++
   Removes columns and rows from the image for augmentation and efficiency.

   ---
   .. link-button:: cutmix.html
      :text: CutMix
      :classes: card-title

   :badge:`CV,badge-primary`
   ++++++++++++++
   Combines pairs of examples in non-overlapping regions and mixes labels

   ---
   .. link-button:: cutout.html
      :text: CutOut
      :classes: card-title

   :badge:`CV,badge-primary`
   ++++++++++++++
   Randomly erases rectangular blocks from the image.

   ---
   .. link-button:: factorize.html
      :text: Factorize
      :classes: card-title

   :badge:`CV,badge-primary` :badge:`NLP,badge-success`
   ++++++++++++++
   Uses Fused LayerNorm kernels for increased GPU utilization.

   ---
   .. link-button:: fused_layernorm.html
      :text: Fused LayerNorm
      :classes: card-title

   :badge:`CV,badge-primary` :badge:`NLP,badge-success`
   ++++++++++++++
   Swaps linear layers for Gated Linear Units in the feed-forward network.

   ---
   .. link-button:: gated_linear_units.html
      :text: Gated Linear Units
      :classes: card-title

   :badge:`NLP,badge-success`
   ++++++++++++++

   Factorize GEMMs into smaller GEMMs

   ---
   .. link-button:: ghost_batchnorm.html
      :text: GhostBatchNorm
      :classes: card-title

   :badge:`CV,badge-primary`
   ++++++++++++++
   Use smaller samples to compute batchnorm

   ---
   .. link-button:: label_smoothing.html
      :text: LabelSmoothing
      :classes: card-title

   :badge:`CV,badge-primary`
   ++++++++++++++
   Smooths the labels with a uniform prior

   ---
   .. link-button:: layer_freezing.html
      :text: LayerFreezing
      :classes: card-title

   :badge:`CV,badge-primary` :badge:`NLP,badge-success`
   ++++++++++++++
   Progressively freezes layers during training.

   ---
   .. link-button:: mixup.html
      :text: MixUp
      :classes: card-title

   :badge:`CV,badge-primary`
   ++++++++++++++
   Blends pairs of examples and labels

   ---
   .. link-button:: progressive_resizing.html
      :text: ProgressiveResizing
      :classes: card-title

   :badge:`CV,badge-primary`
   ++++++++++++++
   Increases the input image size during training

   ---
   .. link-button:: randaugment.html
      :text: RandAugment
      :classes: card-title

   :badge:`CV,badge-primary`
   ++++++++++++++
   Applies a series of random augmentations

   ---
   .. link-button:: sam.html
      :text: SAM
      :classes: card-title

   :badge:`CV,badge-primary`
   ++++++++++++++
   SAM optimizer measures sharpness of optimization space

   ---
   .. link-button:: scale_schedule.html
      :text: ScaleSchedule
      :classes: card-title


   ++++++++++++++
   Scale the learning rate schedule by a factor

   ---
   .. link-button:: selective_backprop.html
      :text: SelectiveBackprop
      :classes: card-title

   :badge:`CV,badge-primary`
   ++++++++++++++
   Drops examples with small loss contributions.

   ---
   .. link-button:: seq_length_warmup.html
      :text: SeqLengthWarmup
      :classes: card-title

   :badge:`NLP,badge-success`
   ++++++++++++++
   Progressively increase sequence length.

   ---
   .. link-button:: squeeze_excite.html
      :text: SqueezeExcite
      :classes: card-title

   :badge:`CV,badge-primary`
   ++++++++++++++
   Replaces eligible layers with Squeeze-Excite layers

   ---
   .. link-button:: stochastic_depth.html
      :text: StochasticDepth
      :classes: card-title

   :badge:`CV,badge-primary`
   ++++++++++++++
   Replaces a specified layer with a stochastic verion that randomly drops the layer or samples during training

   ---
   .. link-button:: swa.html
      :text: SWA
      :classes: card-title

   :badge:`CV,badge-primary` :badge:`NLP,badge-success`
   ++++++++++++++
   Computes running average of model weights.
