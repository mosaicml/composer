<br />
<p align="center">
    <a href="https://github.com/mosaicml/composer#gh-light-mode-only" class="only-light">
      <img src="https://storage.googleapis.com/docs.mosaicml.com/images/header_light.svg" width="50%"/>
    </a>
    <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->
    <a href="https://github.com/mosaicml/composer#gh-dark-mode-only" class="only-dark">
      <img src="https://storage.googleapis.com/docs.mosaicml.com/images/header_dark.svg" width="50%"/>
    </a>
    <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->
</p>

<h2><p align="center">A PyTorch Library for Efficient Neural Network Training</p></h2>
<h3><p align="center">Train Faster, Reduce Cost, Get Better Models</p></h3>

<h4><p align='center'>
<a href="https://www.mosaicml.com">[Website]</a>
- <a href="https://docs.mosaicml.com/en/stable/getting_started/installation.html">[Getting Started]</a>
- <a href="https://docs.mosaicml.com/">[Docs]</a>
- <a href="https://docs.mosaicml.com/en/stable/method_cards/methods_overview.html">[Methods]</a>
- <a href="https://www.mosaicml.com/team">[We're Hiring!]</a>
</p></h4>

<p align="center">
    <a href="https://pypi.org/project/mosaicml/">
        <img alt="PyPi Version" src="https://img.shields.io/pypi/pyversions/mosaicml">
    </a>
    <a href="https://pypi.org/project/mosaicml/">
        <img alt="PyPi Package Version" src="https://img.shields.io/pypi/v/mosaicml">
    </a>
    <a href="https://pepy.tech/project/mosaicml/">
        <img alt="PyPi Downloads" src="https://static.pepy.tech/personalized-badge/mosaicml?period=month&units=international_system&left_color=grey&right_color=blue&left_text=Downloads/month">
    </a>
    <a href="https://docs.mosaicml.com/en/stable/">
        <img alt="Documentation" src="https://readthedocs.org/projects/composer/badge/?version=stable">
    </a>
    <a href="https://join.slack.com/t/mosaicml-community/shared_invite/zt-w0tiddn9-WGTlRpfjcO9J5jyrMub1dg">
        <img alt="Chat @ Slack" src="https://img.shields.io/badge/slack-chat-2eb67d.svg?logo=slack">
    </a>
    <a href="https://github.com/mosaicml/composer/blob/dev/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=slack">
    </a>
</p>
<br />

# 👋 Welcome

Composer is a PyTorch library that enables you to <b>train neural networks faster, at lower cost, and to higher accuracy</b>. We've implemented more than two dozen speedup methods that can be applied to your training loop in just a few lines of code, or used with our built-in Trainer. We continually integrate the latest state-of-the-art in efficient neural network training.

Composer features:
- 20+ methods for speeding up training networks for computer vision and natural language. Don't waste hours trying to reproduce research papers when Composer has done the work for you.
- An easy-to-use trainer that has been written to be as performant as possible and [integrates best practices](https://www.mosaicml.com/blog/5-best-practices-for-efficient-model-training) for efficient, multi-GPU training.
- Functional forms of all of our speedup methods that allow you to integrate them into your existing training loop.
- Strong, reproducible baselines to get you started as quickly as possible.

## Benefits

<!-- start main results -->
<p align="center">
  <a href="https://storage.googleapis.com/docs.mosaicml.com/images/composer_graph_light_06212022.svg?ref=Fiey0Xei#gh-light-mode-only" class="only-light">
    <img src="https://storage.googleapis.com/docs.mosaicml.com/images/composer_graph_light_06212022.svg?ref=Fiey0Xei" width="75%"/>
  </a>
  <!-- link to the light mode image even on dark mode, so it will be readable in a new tab -->
  <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->
  <a href="https://storage.googleapis.com/docs.mosaicml.com/images/composer_graph_dark_06212022.svg?ref=Fiey0Xei#gh-dark-mode-only" class="only-dark">
    <img src="https://storage.googleapis.com/docs.mosaicml.com/images/composer_graph_dark_06212022.svg?ref=Fiey0Xei" width="75%"/>
  </a>
  <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->
</p>
<!-- end main results -->

With no additional tuning, you can apply our methods to:
<!-- start numbers -->
- Train ResNet-50 on ImageNet to the standard 76.6% top-one accuracy for \$15 in 27 minutes (_with vanilla PyTorch:_ \$116 in 3.5 hours) on AWS.
- Train GPT-2 125M to the standard perplexity of 24.11 for \$145 in 4.5 hours (_with vanilla PyTorch_: \$255 in 7.8 hours) on AWS.
- Train DeepLab-v3 on ADE20k to the standard mean IOU of 45.7 for \$36 in 1.1 hours (_with vanilla PyTorch_: \$110 in 3.5 hours) on AWS.
<!-- end numbers -->

# 🚀 Quickstart

## 💾 Installation
Composer is available with Pip:

<!--pytest.mark.skip-->
```bash
pip install mosaicml
```
Alternatively, install Composer with Conda:

<!--pytest.mark.skip-->
```bash
conda install -c mosaicml mosaicml
```
---

## 🚌 Usage

You can use Composer's speedup methods in two ways:

* Through a standalone **Functional API** (similar to `torch.nn.functional`) that allows you to integrate them into your existing training code.
* Using Composer's built-in **Trainer**, which is designed to be performant and automatically takes care of the details of using speedup methods.

### Example: Functional API [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mosaicml/composer/blob/dev/examples/functional_api.ipynb)

Integrate our speedup methods into your training loop with just a few lines of code, and see the results. Here we easily apply [BlurPool](https://docs.mosaicml.com/en/stable/method_cards/blurpool.html) and [SqueezeExcite](https://docs.mosaicml.com/en/stable/method_cards/squeeze_excite.html):

<!-- begin_example_1 --->
```python
import composer.functional as cf
from torchvision import models

my_model = models.resnet18()

# add blurpool and squeeze excite layers
cf.apply_blurpool(my_model)
cf.apply_squeeze_excite(my_model)

# your own training code starts here
```
<!-- end_example_1 --->

For more examples, see the [Composer Functional API Colab notebook](https://colab.research.google.com/github/mosaicml/composer/blob/dev/examples/functional_api.ipynb) and [Functional API guide](https://docs.mosaicml.com/en/latest/functional_api.html).

### Example: Trainer [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mosaicml/composer/blob/dev/examples/getting_started.ipynb)

For the best experience and the most efficient possible training, we recommend using Composer's built-in trainer, which automatically takes care of the details of using speedup methods and provides useful abstractions that facilitate rapid experimentation.

<!-- begin_example_2 --->
<!--pytest.mark.gpu-->
<!--pytest.mark.filterwarnings(r'ignore:Some targets have less than 1 total probability:UserWarning')-->
<!--
```python
import torch

# adaptive_avg_pool2d_backward_cuda in mnist_classifier is not deterministic
torch.use_deterministic_algorithms(False)

```
-->
<!--pytest-codeblocks:cont-->
```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from composer import Trainer
from composer.algorithms import ChannelsLast, CutMix, LabelSmoothing
from composer.models import mnist_model

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST("data", download=True, train=True, transform=transform)
eval_dataset = datasets.MNIST("data", download=True, train=False, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=128)
eval_dataloader = DataLoader(eval_dataset, batch_size=128)

trainer = Trainer(
    model=mnist_model(),
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration="1ep",
    algorithms=[
        ChannelsLast(),
        CutMix(alpha=1.0),
        LabelSmoothing(smoothing=0.1),
    ]
)
trainer.fit()
```
<!-- end_example_2 -->

Composer's built-in [trainer](https://docs.mosaicml.com/en/stable/trainer/using_the_trainer.html) makes it easy to **add multiple speedup methods in a single line of code!**
Trying out new methods or combinations of methods is as easy as changing a single list.

Here are some examples of methods available in Composer ([_see here for the full list_](https://docs.mosaicml.com/en/latest/trainer/algorithms.html)):

Name|Attribution|tl;dr|Example Benchmark|Speed Up*|
----|-----------|-----|---------|---------|
[Alibi](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/alibi)|[Press et al, 2021](https://arxiv.org/abs/2108.12409)|Replace attention with AliBi.|GPT-2|1.5x
[BlurPool](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/blurpool)|[Zhang, 2019](https://arxiv.org/abs/1904.11486)|Applies an anti-aliasing filter before every downsampling operation.|ResNet-101|1.2x
[ChannelsLast](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/channels_last)|[PyTorch](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)|Uses channels last memory format (NHWC).|ResNet-101|1.5x
[CutOut](https://docs.mosaicml.com/en/latest/method_cards/cutout.html)|[DeVries et al, 2017](https://arxiv.org/abs/1708.04552)|Randomly erases rectangular blocks from the image.|ResNet-101|1.2x
[LabelSmoothing](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/label_smoothing)|[Szegedy et al, 2015](https://arxiv.org/abs/1512.00567)|Smooths the labels with a uniform prior|ResNet-101|1.5x
[MixUp](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/mixup)|[Zhang et al, 2017](https://arxiv.org/abs/1710.09412)|Blends pairs of examples and labels.|ResNet-101|1.5x
[RandAugment](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/randaugment)|[Cubuk et al, 2020](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html)|Applies a series of random augmentations to each image.|ResNet-101|1.3x
[SAM](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/sam)|[Foret et al, 2021](https://arxiv.org/abs/2010.01412)|An optimization strategy that seeks flatter minima.|ResNet-101|1.4x
[SeqLengthWarmup](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/seq_length_warmup)|[Li et al, 2021](https://arxiv.org/abs/2108.06084)|Progressively increase sequence length.|GPT-2|1.2x
[Stochastic Depth](https://docs.mosaicml.com/en/latest/method_cards/stochastic_depth.html)|[Huang et al, 2016](https://arxiv.org/abs/1603.09382)|Replaces a specified layer with a stochastic version that randomly drops the layer or samples during training|ResNet-101|1.1x
<p align="right">* = time-to-train to the same quality as the baseline.</p>

## 🛠 Building Speedup Recipes

Given two methods that speed up training by 1.5x each, do they combine to provide a 2.25x (1.5x * 1.5x) speedup? Not necessarily.
They may optimize the [same part of the training process](https://en.wikipedia.org/wiki/Amdahl's_law) and lead to diminishing returns, or they may even interact in ways that prove detrimental.
Determining which methods to compose together isn't as simple as assembling a set of methods that perform best individually.

**We have come up with compositions of methods that work especially well together** through rigorous exploration of the design space of recipes and research on the science behind composition.
The [MosaicML Explorer](https://app.mosaicml.com/) contains all of the data we have collected so far on composition, and it highlights the compositions of methods that are _pareto-optimal_ - that provide the **best possible tradeoffs between training time or cost and the quality of the trained model**.
Whether you want to reach the same quality faster or get better quality within your current budget, Explorer can help you decide which speedup methods to use.
We update this data regularly as we add new methods and develop better recipes.

<p align="center">
<img src="https://storage.googleapis.com/docs.mosaicml.com/images/methods/explorer.png"/>
</p>

As an example, here are two performant recipes, one for ResNet-101 on ImageNet, and the other for GPT-2 on OpenWebText, on 8xA100s:

### ResNet-101

Name|Functional|tl;dr|Benchmark|Speed Up
----|----------|-----|---------|--------
[Blur Pool](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/blurpool)|`cf.apply_blurpool`|[Applies an anti-aliasing filter before every downsampling operation.](https://arxiv.org/abs/1904.11486)|ResNet-101|1.2x
[Channels Last](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/channels_last)|`cf.apply_`<br>`channels_last`|[Uses channels last memory format (NHWC).](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)|ResNet-101|1.5x
[Label Smoothing](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/label_smoothing)|`cf.smooth_labels`|[Smooths the labels with a uniform prior.](https://arxiv.org/abs/1512.00567)|ResNet-101|1.5x
[MixUp](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/mixup)|`CF.mixup_batch`|[Blends pairs of examples and labels.](https://arxiv.org/abs/1710.09412)|ResNet-101|1.5x
[Progressive Resizing](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/progressive_resizing)|`cf.resize_batch`|[Increases the input image size during training.](https://github.com/fastai/fastbook/blob/780b76bef3127ce5b64f8230fce60e915a7e0735/07_sizing_and_tta.ipynb)|ResNet-101|1.3x
[SAM](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/sam)|`N/A`|[SAM optimizer measures sharpness of optimization space.](https://arxiv.org/abs/2010.01412)|ResNet-101|1.5x
**Composition** | `N/A` | **Cheapest: \$49 @ 78.1% Acc** | **ResNet-101** | **3.5x**

### GPT-2

Name|Functional|tl;dr|Benchmark|Speed Up
----|----------|-----|---------|--------
[Alibi](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/alibi)|`cf.apply_alibi`|[Replace attention with AliBi.](https://arxiv.org/abs/2108.12409)|GPT-2|1.6x
[Seq Length Warmup](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/seq_length_warmup)|`cf.set_batch_`<br>`sequence_length`|[Progressively increase sequence length.](https://arxiv.org/abs/2108.06084)|GPT-2|1.5x
**Composition** | `N/A` | **Cheapest: \$145 @ 24.11 PPL** | **GPT-2** | **1.7x**

# ⚙️ What benchmarks does Composer support?

We'll use the word _benchmark_ to denote a specific model trained on a specific dataset, with model quality assessed using a specific metric.

Composer features computer vision and natural language processing benchmarks including (but not limited to):

<div class="center">
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Dataset</th>
    <th>Loss</th>
    <th>Task</th>
    <th>Evaluation Metrics</th>
  </tr>
</thead>
<tbody>
  <tr>
      <td colspan="5" align="center"><b>Computer Vision</b></td>
  </tr>
  <tr>
    <td>ResNet Family</td>
    <td>CIFAR-10</td>
    <td>Cross Entropy</td>
    <td>Image Classification</td>
    <td>Classification Accuracy</td>
  </tr>
  <tr>
    <td>ResNet Family</td>
    <td>ImageNet</td>
    <td>Cross Entropy</td>
    <td>Image Classification</td>
    <td>Classification Accuracy</td>
  </tr>
  <tr>
    <td>EfficientNet Family</td>
    <td>ImageNet</td>
    <td>Cross Entropy</td>
    <td>Image Classification</td>
    <td>Classification Accuracy</td>
  </tr>
  <tr>
    <td>UNet</td>
    <td>BraTS</td>
    <td>Dice Loss</td>
    <td>Image Segmentation</td>
    <td>Dice Coefficient</td>
  </tr>
  <tr>
    <td>DeepLab v3</td>
    <td>ADE20K</td>
    <td>Cross Entropy</td>
    <td>Image Segmentation</td>
    <td>mIoU</td>
  </tr>
  <tr>
      <td align="center" colspan="5"><b>Natural Language Processing</b></td>
  </tr>
  <tr>
    <td>BERT Family</td>
    <td>{Wikipedia &amp; BooksCorpus, C4}</td>
    <td>Cross Entropy</td>
    <td>Masked Language Modeling</td>
    <td>GLUE </td>
  </tr>
  <tr>
    <td>GPT Family</td>
    <td>{OpenWebText, C4}</td>
    <td>Cross Entropy</td>
    <td>Language Modeling<br></td>
    <td>Perplexity</td>
  </tr>
</tbody>
</table>
</div>

# 🤔 Why should I use Composer?

### Speed

The compute required to train a state-of-the-art machine learning model is [doubling every 6 months](https://arxiv.org/abs/2202.05924), putting such models further and further out of reach for most researchers and practitioners with each passing day.

Composer addresses this challenge by focusing on training efficiency: it contains cutting-edge speedup methods that modify the training algorithm to reduce the time and cost necessary to train deep learning models.
**When you use Composer, you can rest assured that you are training efficiently.**
We have combed the literature, done the science, and built industrial-grade implementations to ensure this is the case.

### Flexibility

Even after these speedup methods are implemented, assembling them together into recipes is nontrivial. We designed Composer with the **right abstractions for composing (and creating new) speedup methods.**

Specifically, Composer uses two-way callbacks ([Howard et al, 2020](https://arxiv.org/abs/2002.04688)) to modify the **entire training state** at particular events in the training loop to effect speedups. We handle collisions between methods, proper method ordering, and more.

Through this, methods can modify:
 - data inputs for batches (data augmentations, sequence length warmup, skipping examples, etc.)
 - neural network architecture (pruning, model surgery, etc.)
 - loss function (label smoothing, MixUp, CutMix, etc.)
 - optimizer (Sharpness Aware Minimization)
 - training dynamics (layer freezing, selective backprop, etc.)

You can easily [add your own methods](https://colab.research.google.com/github/mosaicml/composer/blob/dev/examples/custom_speedup_methods.ipynb) or callbacks to try out your ideas or modify any part of the training loop.

### Support

Composer is an active and ongoing project. We will respond quickly to issues filed in this repository.

# 🧐 Why shouldn’t I use Composer?

* Composer is mostly optimized for computer vision and natural language processing. If you work on, e.g., reinforcement learning, you might encounter rough edges when using Composer.
* Composer currently only supports NVIDIA GPUs, although we're working on adding alternatives.
* Since Composer is still in alpha, our API may not be stable. We recommend pegging your work to a Composer version.

# 📚 Learn More

Here are some resources actively maintained by the Composer community to help you get started:
<table>
<thead>
  <tr>
      <th><b>Resource</b></th>
      <th><b>Details</b></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="https://colab.research.google.com/github/mosaicml/composer/blob/dev/examples/getting_started.ipynb" target="_blank" rel="noopener noreferrer">Getting started with our Trainer</a></td>
    <td>A Colab Notebook showing how to use our Trainer</td>
  </tr>
  <tr>
    <td><a href="https://colab.research.google.com/github/mosaicml/composer/blob/dev/examples/functional_api.ipynb" target="_blank" rel="noopener noreferrer">Getting started with our Functional API</a></td>
    <td>A Colab Notebook showing how to use our Functional API</td>
  </tr>
  <tr>
    <td><a href="https://colab.research.google.com/github/mosaicml/composer/blob/dev/examples/custom_speedup_methods.ipynb" target="_blank" rel="noopener noreferrer">Building Speedup Methods</a></td>
    <td>A Colab Notebook showing how to build new training modifications on top of Composer</td>
  </tr>
  <tr>
    <td><a href="https://colab.research.google.com/github/mosaicml/composer/blob/dev/examples/finetune_huggingface.ipynb" target="_blank" rel="noopener noreferrer">Training BERTs with Composer and 🤗 </a></td>
    <td>A Colab Notebook showing how to train BERT models with Composer and 🤗!</td>
  </tr>
</tbody>
</table>

If you have any questions, please feel free to reach out to us on [Twitter](https://twitter.com/mosaicml), [email](mailto:community@mosaicml.com), or our [Community Slack](https://join.slack.com/t/mosaicml-community/shared_invite/zt-1dc6mo5wg-arlv6Oo9JjEn_g4d5s7PXQ)!

# 💫 Contributors
Composer is part of the broader Machine Learning community, and we welcome any contributions, pull requests, or issues!

To start contributing, see our [Contributing](https://github.com/mosaicml/composer/blob/dev/CONTRIBUTING.md) page.

P.S.: [We're hiring](https://mosaicml.com/jobs)!

# ✍️ Citation
```
@misc{mosaicml2022composer,
    author = {The Mosaic ML Team},
    title = {composer},
    year = {2021},
    howpublished = {\url{https://github.com/mosaicml/composer/}},
}
```
