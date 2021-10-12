# Methods

Each method is implemented as a two-way callback for composability, and also in functional form for standalone access. For example, `BlurPool` can be used either with:

```python
import composer.functional as CF

my_model = CF.apply_blurpool(my_model)
```

or as a method used with our engine:

```python
from composer import Engine, Event, algorithms

engine = Engine(state, algorithms=[algorithms.BlurPool()])

engine.run_algorithms(event=Event.TRAINING_START)  # applies blurpool to state.model
```
<!--- Table Start (below is auto generated) -->
Name|Functional|Attribution|tl;dr|Card
----|----------|-----------|-----|----
[AugMix](https://github.com/mosaicml/mosaicml/tree/dev/composer/algorithms/augmix)|`CF.augment_and_mix`|[(Hendrycks et al, 2020)](http://arxiv.org/abs/1912.02781)||
[BlurPool](https://github.com/mosaicml/mosaicml/tree/dev/composer/algorithms/blurpool)|`CF.apply_blurpool`|[(Zhang, 2019)](https://arxiv.org/abs/1904.11486)||
[ChannelsLast](https://github.com/mosaicml/mosaicml/tree/dev/composer/algorithms/channels_last)||[PyTorch](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)|Uses channels last memory format (NHWC)|
[ColOut](https://github.com/mosaicml/mosaicml/tree/dev/composer/algorithms/colout)|`CF.colout`|MosaicML|Removes columns and rows from the image for augmentation and efficiency.|
[CutOut](https://github.com/mosaicml/mosaicml/tree/dev/composer/algorithms/cutout)|`CF.cutout`|[(DeVries et al, 2017)](https://arxiv.org/abs/1708.04552)|Randomly erases rectangular blocks from the image.|
[LabelSmoothing](https://github.com/mosaicml/mosaicml/tree/dev/composer/algorithms/label_smoothing)|`CF.smooth_labels`|[(Szegedy et al, 2015)](https://arxiv.org/abs/1512.00567)|Smooths the labels with a uniform prior|
[LayerFreezing](https://github.com/mosaicml/mosaicml/tree/dev/composer/algorithms/layer_freezing)|`CF.freeze_layers`|[Many (Raghu et al, 2017)](https://arxiv.org/abs/1706.05806)|Progressively freezes layers during training.|
[MixUp](https://github.com/mosaicml/mosaicml/tree/dev/composer/algorithms/mixup)|`CF.mixup_batch`|[(Zhang et al, 2017)](https://arxiv.org/abs/1710.09412)|Blends pairs of examples and labels|
[NoOpModel](https://github.com/mosaicml/mosaicml/tree/dev/composer/algorithms/no_op_model)||MosaicML|Replaces the model with a dummy model. Used for dataloader profiling.|
[ProgressiveResizing](https://github.com/mosaicml/mosaicml/tree/dev/composer/algorithms/progressive_resizing)|`CF.resize_inputs`|[Fast AI](https://github.com/fastai/fastbook/blob/780b76bef3127ce5b64f8230fce60e915a7e0735/07_sizing_and_tta.ipynb)|Increases the input image size during training|
[RandAugment](https://github.com/mosaicml/mosaicml/tree/dev/composer/algorithms/randaugment)|`CF.randaugment`|[(Cubuk et al, 2020)](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html)|Applies a series of random augmentations|
[SAM](https://github.com/mosaicml/mosaicml/tree/dev/composer/algorithms/sam)||[(Foret et al, 2021)](https://arxiv.org/abs/2010.01412)|SAM optimizer measures sharpness of optimization space|
[ScaleScheduler](https://github.com/mosaicml/mosaicml/tree/dev/composer/algorithms/scale_schedule)|`CF.scale_scheduler`||Scales the learning schedule by a factor|
[SqueezeExcite](https://github.com/mosaicml/mosaicml/tree/dev/composer/algorithms/squeeze_excite)|`CF.apply_se`|[Hu et al, 2017](https://arxiv.org/abs/1709.01507)|Replaces eligible layers with Squeeze-Excite layers|
<!--- Table End -->

## Contributing
<TODO> update this!

To contribute an algorithm, use the provided `template.py` in this folder and modify accordingly. Create a new folder `algorithms/my_algorithm` and register the algorithm in `algorithm_registry.py`:
```
<TODO> add runtime code to do this
```

Your algorithm should now be accessible via our argparse interface:

```python
python examples/run_mosaicml_trainer.py -f model.yaml --algorithms my_algorithm --algorithms.my_algorithm.alpha 0.1
```

or through inclusion in our `yaml` file:

```python
algorithms:
  - my_algorithm
    - alpha: 0.1
```

or directly importing the module for developers into their own trainers:

```
from composer.algorithms import MyAlgorithm
algo = MyAlgorithm(alpha=0.1)

```

If you algorithm has specific requirements, please add them to `setup.py` in your PR.

### Testing

For testing, remember to:

* add your own tests to `tests/algorithms/test_my_algorithm.py`
* add the algorithm defaults for testing into `tests/algorithms/test_algorithm_registry.py`
* add your docstrings

Note that `tests/fixtures/dummy_fixtures.py` contains some fixtures such as a simple conv model which are useful for writing tests.
