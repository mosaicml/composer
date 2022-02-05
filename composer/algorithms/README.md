# Methods

Each method is implemented as a two-way callback for composability, and also in functional form for standalone access. For example, `BlurPool` can be used either with:

```python
from composer import functional as CF

my_model = CF.apply_blurpool(my_model)
```

or as a method used with our engine:

```python
from composer import Engine, Event, algorithms

engine = Engine(state, algorithms=[algorithms.BlurPool()])

engine.run_algorithms(event=Event.INIT)  # applies blurpool to state.model
```
<!--- Table Start (below is auto generated) -->
Name|Functional|Attribution|tl;dr|Card
----|----------|-----------|-----|----
[Alibi](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/alibi)|`CF.apply_alibi`|[(Press et al, 2021)](https://arxiv.org/abs/2108.12409v1)|Replace attention with AliBi|
[AugMix](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/augmix)|`CF.augment_and_mix`|[(Hendrycks et al, 2020)](http://arxiv.org/abs/1912.02781)|Image-perserving data augmentations|
[BlurPool](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/blurpool)|`CF.apply_blurpool`|[(Zhang, 2019)](https://arxiv.org/abs/1904.11486)||
[ChannelsLast](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/channels_last)||[PyTorch](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)|Uses channels last memory format (NHWC)|
[ColOut](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/colout)|`CF.colout`|MosaicML|Removes columns and rows from the image for augmentation and efficiency.|
[SeqLengthWarmup](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/seq_length_warmup)|`CF.apply_seq_length_warmup`|[(Li et al, 2021)](https://arxiv.org/abs/2108.06084)|Progressively increase sequence length.|
[CutOut](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/cutout)|`CF.cutout`|[(DeVries et al, 2017)](https://arxiv.org/abs/1708.04552)|Randomly erases rectangular blocks from the image.|
[GhostBatchNorm](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/ghost_batchnorm)|`CF.apply_ghost_batchnorm`|[(Dimitriou et al, 2020)](https://arxiv.org/abs/2007.08554)|Use smaller samples to compute batchnorm|
[LabelSmoothing](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/label_smoothing)|`CF.smooth_labels`|[(Szegedy et al, 2015)](https://arxiv.org/abs/1512.00567)|Smooths the labels with a uniform prior|
[LayerFreezing](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/layer_freezing)|`CF.freeze_layers`|[Many (Raghu et al, 2017)](https://arxiv.org/abs/1706.05806)|Progressively freezes layers during training.|
[MixUp](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/mixup)|`CF.mixup_batch`|[(Zhang et al, 2017)](https://arxiv.org/abs/1710.09412)|Blends pairs of examples and labels|
[NoOpModel](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/no_op_model)||MosaicML|Replaces the model with a dummy model. Used for dataloader profiling.|
[ProgressiveResizing](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/progressive_resizing)|`CF.resize_inputs`|[Fast AI](https://github.com/fastai/fastbook/blob/780b76bef3127ce5b64f8230fce60e915a7e0735/07_sizing_and_tta.ipynb)|Increases the input image size during training|
[RandAugment](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/randaugment)|`CF.randaugment`|[(Cubuk et al, 2020)](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html)|Applies a series of random augmentations|
[SAM](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/sam)||[(Foret et al, 2021)](https://arxiv.org/abs/2010.01412)|SAM optimizer measures sharpness of optimization space|
[ScaleScheduler](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/scale_schedule)|`CF.scale_scheduler`||Scales the learning schedule by a factor|
[SelectiveBackprop](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/selective_backprop)|`CF.selective_backprop`|[(Jiang et al, 2019)](https://arxiv.org/abs/1910.00762)|Drops examples with small loss contributions.|
[SqueezeExcite](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/squeeze_excite)|`CF.apply_se`|[Hu et al, 2017](https://arxiv.org/abs/1709.01507)|Replaces eligible layers with Squeeze-Excite layers|
[StochasticDepth](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/stochastic_depth)|`CF.apply_stochastic_depth`|[(Huang et al, 2016)](https://arxiv.org/abs/1603.09382)|Replaces a specified layer with a stochastic verion that randomly drops the layer or samples during training|
[SWA](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/swa)||[(Izmailov et al, 2018)](https://arxiv.org/abs/1803.05407)|Computes running average of model weights.|
<!--- Table End -->


