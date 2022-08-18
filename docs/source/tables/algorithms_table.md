```{eval-rst}
:orphan:
```

Name|Functional|Attribution|tl;dr|Math-Equivalent
----|----------|-----------|-----|----
[Alibi](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/alibi)|`cf.apply_alibi`|[Press et al, 2021](https://arxiv.org/abs/2108.12409v1)|Replace attention with AliBi|❌
[AugMix](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/augmix)|`cf.augmix_image`|[Hendrycks et al, 2020](http://arxiv.org/abs/1912.02781)|Image-perserving data augmentations|❌
[BlurPool](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/blurpool)|`cf.apply_blurpool`|[Zhang, 2019](https://arxiv.org/abs/1904.11486)|applies blur before pooling|❌
[ChannelsLast](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/channels_last)|`cf.apply_channels_last`|[PyTorch](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)|Uses channels last memory format (NHWC)|✅
[ColOut](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/colout)|`cf.colout_batch`|[MosaicML](https://www.mosaicml.com)|Removes columns and rows from the image for augmentation and efficiency|❌
[CutMix](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/cutmix)|`cf.cutmix_batch`|[Yun et al, 2019](https://arxiv.org/abs/1905.04899)|Combines pairs of examples in non-overlapping regions and mixes labels|❌
[CutOut](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/cutout)|`cf.cutout_batch`|[DeVries et al, 2017](https://arxiv.org/abs/1708.04552)|Randomly erases rectangular blocks from the image|❌
[Decoupled Weight Decay](https://github.com/mosaicml/composer/blob/dev/composer/optim/decoupled_weight_decay.py)||[Loshchilov and Hutter, 2019](https://arxiv.org/abs/1711.05101)|Decouples weight decay from learning rate|❌
[EMA](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/ema)|`cf.compute_ema`|[Many](https://www.mosaicml.com)|Exponentially weighted moving average of model parameters during training|❌
[Factorize](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/factorize)|`cf.apply_factorization`|[Many](https://www.mosaicml.com)|Factorize GEMMs into smaller GEMMs|❌
[Fused LayerNorm](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/fused_layernorm)|`cf.apply_fused_layernorm`|[Many](https://www.mosaicml.com)|Replaces `torch.nn.LayerNorm` with `apex.normalization.fused_layer_norm`|✅
[Gated Linear Units](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/gated_linear_units)|`cf.apply_gated_linear_units`|[Dauphin et al., 2016](https://arxiv.org/abs/1612.08083)|Adds gating operation to the input of feed-forward block.|❌
[GhostBatchNorm](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/ghost_batchnorm)|`cf.apply_ghost_batchnorm`|[Dimitriou et al, 2020](https://arxiv.org/abs/2007.08554)|Use smaller samples to compute batchnorm|❌
[LabelSmoothing](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/label_smoothing)|`cf.smooth_labels`|[Szegedy et al, 2015](https://arxiv.org/abs/1512.00567)|Smooths the labels with a uniform prior|❌
[LayerFreezing](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/layer_freezing)|`cf.freeze_layers`|[Many; Raghu et al, 2017](https://arxiv.org/abs/1706.05806)|Progressively freezes layers during training|❌
[MixUp](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/mixup)|`cf.mixup_batch`|[Zhang et al, 2017](https://arxiv.org/abs/1710.09412)|Blends pairs of examples and labels|❌
[ProgressiveResizing](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/progressive_resizing)|`cf.resize_batch`|[Fast AI](https://github.com/fastai/fastbook/blob/780b76bef3127ce5b64f8230fce60e915a7e0735/07_sizing_and_tta.ipynb)|Scale the input image size over the course of training|❌
[RandAugment](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/randaugment)|`cf.randaugment_image`|[Cubuk et al, 2020](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html)|Applies a series of random augmentations|❌
[SAM](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/sam)||[Foret et al, 2021](https://arxiv.org/abs/2010.01412)|SAM optimizer measures sharpness of optimization space|❌
[ScaleSchedule](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/scale_schedule)||[Many](https://www.mosaicml.com)|Scale the learning rate schedule by a factor|❌
[SelectiveBackprop](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/selective_backprop)|`cf.selective_backprop`|[Jiang et al, 2019](https://arxiv.org/abs/1910.00762)|Drops examples with small loss contributions|❌
[SeqLengthWarmup](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/seq_length_warmup)|`cf.set_batch_sequence_length`|[Li et al, 2021](https://arxiv.org/abs/2108.06084)|Progressively increase sequence length|❌
[SqueezeExcite](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/squeeze_excite)|`cf.apply_squeeze_excite`|[Hu et al, 2017](https://arxiv.org/abs/1709.01507)|Replaces eligible layers with Squeeze-Excite layers|❌
[StochasticDepth](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/stochastic_depth)|`cf.apply_stochastic_depth`|[Huang et al, 2016](https://arxiv.org/abs/1603.09382)|Replaces a specified layer with a stochastic verion that randomly drops the layer or samples during training|❌
[SWA](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/swa)||[Izmailov et al, 2018](https://arxiv.org/abs/1803.05407)|Computes running average of model weights|❌
