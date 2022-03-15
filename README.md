<br /><br />
<p align="center">
    <a href="https://github.com/mosaicml/composer#gh-light-mode-only" class="only-light">
      <img src="https://storage.googleapis.com/docs.mosaicml.com/images/header_light.svg" width="70%"/>
    </a>
    <a href="https://github.com/mosaicml/composer#gh-dark-mode-only" class="only-dark">
      <img src="https://storage.googleapis.com/docs.mosaicml.com/images/header_dark.svg#gh-dark-mode-only" class="only-dark" width="70%"/>
    </a>
</p>
<br />

<h1><p align="center">A PyTorch Library for Efficient Neural Network Training</p></h1>
<h2><p align="center">Train Faster, Reduce Cost, Get Better Models</p></h2>

<h3><p align='center'>
<a href="https://www.mosaicml.com">[Website]</a>
- <a href="https://docs.mosaicml.com/en/stable/getting_started/installation.html">[Getting Started]</a>
- <a href="https://docs.mosaicml.com/">[Docs]</a>
- <a href="https://docs.mosaicml.com/en/stable/method_cards/methods_overview.html">[Methods]</a>
- <a href="https://www.mosaicml.com/team">[We're Hiring!]</a>
</p></h3>

<p align="center">
    <a href="https://pypi.org/project/mosaicml/">
        <img alt="PyPi Version" src="https://img.shields.io/pypi/pyversions/mosaicml">
    </a>
    <a href="https://pypi.org/project/mosaicml/">
        <img alt="PyPi Package Version" src="https://img.shields.io/pypi/v/mosaicml">
    </a>
    <a href="https://pypi.org/project/mosaicml/">
        <img alt="PyPi Downloads" src="https://img.shields.io/pypi/dm/mosaicml">
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

Composer is a library written in PyTorch that makes it possible to <b>train neural networks faster, at lower cost, and to higher accuracy or perplexity</b>.
It contains implementations of standard neural network benchmarks like ResNets for image classification, GPT and BERT models for language modeling, DeepLab and UNet for segmentation, and SSD for object detection.

<p align="center">
<img src="https://storage.googleapis.com/docs.mosaicml.com/images/cost_graph_light.svg#gh-light-mode-only" class="only-light" width="85%"/>
<img src="https://storage.googleapis.com/docs.mosaicml.com/images/cost_graph_dark.svg#gh-dark-mode-only" class="only-dark" width="85%"/>
</p>

Composer is designed to train these benchmarks and your custom benchmarks <b>as efficiently as possible</b>.
In addition to <b>carefully tuned implementations of all aspects of the training process</b>, Composer includes <b>more than two dozen speedup methods</b> that modify the training algorithm to (1) reduce the time necessary to train models to standard _quality_ (i.e., accuracy, perplexity, or other metrics of interest) and (2) increase the quality of models trained for a standard amount of time.
You can use these methods through Composer's built-in training loop or apply them to your existing codebase.

In summary, Composer features:
- 20+ methods for speeding up training networks for computer vision and language modeling. Don't waste hours trying to reproduce research papers when Composer has done the work for you.
- An easy-to-use trainer that has been written to be as performant as possible and [integrates best practices](https://www.mosaicml.com/blog/5-best-practices-for-efficient-model-training) for efficient training.
- Functional forms of all of our speedup methods that allow you to integrate them into your existing training loop.
- Strong, _reproducible_ baselines to get you started as quickly as possible.

## Benefits

Composer offers significant reductions in training time and cost. Using Composer, you can:

- Train ResNet-50 on ImageNet to the standard 76.6% top-one accuracy for \$40 in 1 hour and 14 minutes (_with vanilla PyTorch:_ \$116 in 3 hours and 53 minutes) on AWS.
- Train a GPT-2 125M to a standard perplexity of 24.11 for \$145 in 4 hours and 27 minutes (_with vanilla PyTorch_: \$255 in 7 hours and 47 minutes) on AWS.
- Use these speedup methods on your own models in your own training loop.

## The Latest Research on Efficient Training

At MosaicML, our goal is to make training state-of-the-art deep learning models accessible to everyone.
We do so through Composer:

* We continually integrate <b>state-of-the-art research on efficient neural network training</b> (and our improvements to that research).
* These speedup methods work on our <b>efficient implementations of up-to-date models and datasets</b>.
* We provide <b>curated recipes of speedup methods</b> that lead to the most efficient possible training on these benchmarks.


# 🚀 Quickstart

## 💾 Installation
Composer is available with Pip:

```bash
pip install mosaicml
```
Alternatively, install Composer with Conda:

```bash
conda install -c mosaicml mosaicml
```
---

## 🚌 Usage

You can use Composer's speedup methods in two ways:

* Through a standalone **Functional API** (similar to `torch.nn.functional`) that allows you to integrate them into your existing training code.
* Using Composer's built-in **Trainer**, which is designed to be performant and automatically takes care of many of the low-level details of using speedup methods.

### Example: Functional API [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mosaicml/composer/blob/dev/notebooks/Composer_Functional.ipynb)

Integrate our speed-up methods into your training loop with just a few lines of code, and see the results. Here we easily apply [BlurPool](https://docs.mosaicml.com/en/stable/method_cards/blurpool.html) and SqueezeExcite:


```python
import composer.functional as cf
from torchvision import models

my_model = models.resnet18()

# add blurpool and squeeze excite layers
model = cf.apply_blurpool(my_model)
model = cf.apply_squeeze_excite(my_model)

# your own training code starts here
```

For more examples, see the [Composer Functional API Colab notebook](https://colab.research.google.com/github/mosaicml/composer/blob/dev/notebooks/Composer_Functional.ipynb) and [Functional API guide](https://docs.mosaicml.com/en/latest/functional_api.html).

### Example: Trainer [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mosaicml/composer/blob/dev/notebooks/up_and_running_with_composer.ipynb)

For the best experience and the most efficient possible training, we recommend using Composer's built-in trainer, which automatically takes care of the low-level details of using speedup methods and provides useful abstractions that facilitate rapid experimentation.

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from composer import Trainer
from composer.algorithms import BlurPool, ChannelsLast, CutMix, LabelSmoothing
from composer.models import MNIST_Classifier

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST("data", download=True, transform=transform)
train_dataloader = DataLoader(dataset, batch_size=128)

trainer = Trainer(
    model=MNIST_Classifier(num_classes=10),
    train_dataloader=train_dataloader,
    max_duration="2ep",
    algorithms=[
        BlurPool(replace_convs=True, replace_maxpools=True, blur_first=True),
        ChannelsLast(),
        CutMix(num_classes=10),
        LabelSmoothing(smoothing=0.1),
    ]
)
trainer.fit()
```

Composer's built-in trainer makes it easy to **add multiple speedup methods in a single line of code!** 
Trying out new methods or combinations of methods is as easy as changing a single list.
As we continually implement more methods, they will be easy for you to add to your code.

For concrete examples of methods in Composer, here are some ([_see here for all_](https://docs.mosaicml.com/en/latest/method_cards/methods_overview.html)) speedup methods currently in Composer:

Name|Functional|Attribution|tl;dr|Example Benchmark|Speed Up*|
----|----------|-----------|-----|---------|---------|
[Alibi](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/alibi)|`cf.apply_alibi`|[Press et al, 2021](https://arxiv.org/abs/2108.12409v1)|Replace attention with AliBi.|GPT-2|1.5x
[BlurPool](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/blurpool)|`cf.apply_blurpool`|[Zhang, 2019](https://arxiv.org/abs/1904.11486)|Applies an anti-aliasing filter before every downsampling operation.|ResNet-101|1.2x
[ChannelsLast](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/channels_last)|`cf.apply_channels_last`|[PyTorch](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)|Uses channels last memory format (NHWC).|ResNet-101|1.5x
[CutOut](https://docs.mosaicml.com/en/latest/method_cards/cutout.html)|`CF.cutout_batch`|[DeVries et al, 2017](https://arxiv.org/abs/1708.04552)|Randomly erases rectangular blocks from the image.|ResNet-101|1.2x
[LabelSmoothing](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/label_smoothing)|`cf.smooth_labels`|[Szegedy et al, 2015](https://arxiv.org/abs/1512.00567)|Smooths the labels with a uniform prior|ResNet-101|1.5x
[MixUp](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/mixup)|`cf.mixup_batch`|[Zhang et al, 2017](https://arxiv.org/abs/1710.09412)|Blends pairs of examples and labels.|ResNet-101|1.5x
[RandAugment](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/randaugment)|`cf.randaugment_image`|[Cubuk et al, 2020](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html)|Applies a series of random augmentations to each image.|ResNet-101|1.3x
[SAM](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/sam)|`N/A` (Composer trainer only)|[Foret et al, 2021](https://arxiv.org/abs/2010.01412)|An optimization strategy that seeks flatter minima.|ResNet-101|1.4x
[SeqLengthWarmup](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/seq_length_warmup)|`cf.set_batch_sequence_length`|[Li et al, 2021](https://arxiv.org/abs/2108.06084)|Progressively increase sequence length.|GPT-2|1.2x
[Stochastic Depth](https://docs.mosaicml.com/en/latest/method_cards/stochastic_depth.html)|`CF.apply_`<br>`stochastic_depth`|[Huang et al, 2016](https://arxiv.org/abs/1603.09382)|Replaces a specified layer with a stochastic version that randomly drops the layer or samples during training|ResNet-101|1.1x
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
[Alibi](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/alibi)|`cf.apply_alibi`|[Replace attention with AliBi.](https://arxiv.org/abs/2108.12409v1)|GPT-2|1.6x
[Seq Length Warmup](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/seq_length_warmup)|`cf.set_batch_`<br>`sequence_length`|[Progressively increase sequence length.](https://arxiv.org/abs/2108.06084)|GPT-2|1.5x
**Composition** | `N/A` | **Cheapest: \$145 @ 24.11 PPL** | **GPT-2** | **1.7x**

# ⚙️ What benchmarks does Composer support?

Composer uses a _benchmark_ as a term to denote a particular model trained on a particular dataset in a standardized, reproducible way. A benchmark is a specific model trained for a task, where a task is defined as a dataset with a specific loss function.

Composer is currently focused on supporting computer vision and natural language processing use cases, such as (but not limited to) the following. New benchmarks will be added regularly, as will compatibility with existing libraries of benchmarks!

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

The compute required to train a state-of-the-art machine learning model is [doubling every 6 months](https://arxiv.org/abs/2202.05924), putting state-of-the-art capabilities further and further out of reach for the broader community with each passing day.
Composer is designed to address this challenge by focusing on training efficiency: it contains cutting-edge speedup methods that modify the training algorithm to reduce the time and cost necessary to train deep learning models.
**When you use Composer, you can rest assured that you are training efficiently.**
We have combed the literature, done the science, and built industrial-grade implementations to ensure this is the case.

Even once these speedup methods are implemented, composing them together into recipes is a nontrivial task.
We have found that existing training libraries do not provide the capabilities necessary to intervene in the training process in the ways that are necessary to easily swap speedup methods in and out.
To fulfill this need, **we have designed Composer with the right abstractions to make it easy to use (and create new) speedup methods.**

Specifically, Composer provides the ability to use **two-way callbacks** ([Howard et al, 2020](https://arxiv.org/abs/2002.04688)) that are activated when specific events take place during training.
This makes it easy to inject speedup methods in exactly the right places without cluttering the main training loop. 
Each callback has access to the **entire training state** so that it can make whatever changes are necessary for the speedup method to do its job.

Composer leverages the two-way callback system to integrate our methods into the appropriate parts of the training pipeline:
 - Speedup methods can modify data inputs for batches (data augmentations, sequence length warmup, skipping examples, etc)
 - Speedup methods can modify the neural network architecture (layer freezing, pruning, model surgery, etc)
 - Speedup methods can modify the loss function (label smoothing, MixUp, CutMix, etc)
 - Speedup methods can modify the optimizer (Sharpness Aware Minimization)

Composer makes it easy to [add your own methods](https://colab.research.google.com/github/mosaicml/composer/blob/dev/notebooks/custom_method_tutorial.ipynb) or callbacks to instrument any part of the training loop.

# 🧐 Why shouldn’t I use Composer?

* Composer currently supports standard computer vision (CV) and natural language processing (NLP) use cases, both through the included models and datasets and those that you can add yourself. If your use case has specific needs for how the data, model, or loss is handled, Composer may not yet support it. We will continue to add the capabilities to support an increasingly diverse array of settings, but it may currently lack support for needs specific to certain use cases.
* Composer currently supports NVIDIA GPUs. If you use other hardware platforms, Composer cannot currently support your use case without modification. We are adding support for additional hardware platforms, and you should expect more soon.
* Composer is an active and ongoing project. Since Composer is still in alpha, our API may not be stable and it likely has bugs. We recommend pegging your work to a Composer version, and we will respond quickly to issues posted to this repository.

# 📚 Learn More

Here's some resources actively maintained by the Composer community to help you get started:
<table>
<thead>
  <tr>
      <th><b>Resource</b></th>
      <th><b>Details</b></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="https://www.mosaicml.com/blog/making-ml-training-efficient-algorithmically" target="_blank" rel="noopener noreferrer">Founder's Blog</a></td>
    <td>A blog post by our founders highlighting why MosaicML exists</td>
  </tr>
  <tr>
    <td><a href="https://www.mosaicml.com/blog/methodology" target="_blank" rel="noopener noreferrer">Methodology Blog</a></td>
    <td>A blog post describing our scientific methodology for creating and evaluating speedups</td>
  </tr>
  <tr>
    <td><a href="https://colab.research.google.com/github/mosaicml/composer/blob/dev/notebooks/up_and_running_with_composer.ipynb" target="_blank" rel="noopener noreferrer">Getting started with our Trainer</a></td>
    <td>An interactive Colab Notebook aimed at teaching users about our Trainer</td>
  </tr>
  <tr>
    <td><a href="https://colab.research.google.com/github/mosaicml/composer/blob/dev/notebooks/Composer_Functional.ipynb" target="_blank" rel="noopener noreferrer">Getting started with our Functional API</a></td>
    <td>An interactive Colab Notebook aimed at teaching users about our Functional API</td>
  </tr>
  <tr>
    <td><a href="https://colab.research.google.com/github/mosaicml/composer/blob/dev/notebooks/custom_method_tutorial.ipynb" target="_blank" rel="noopener noreferrer">Building Speedup Methods</a></td>
    <td>An interactive Colab Notebook aimed at teaching users about building speedup methods on top of Composer</td>
  </tr>
  <tr>
    <td><a href="https://colab.research.google.com/github/mosaicml/composer/blob/dev/notebooks/nlp_notebook_tutorial.ipynb" target="_blank" rel="noopener noreferrer">Training BERTs with Composer</a></td>
    <td>An interactive Colab Notebook aimed at helping users learn how to train BERT models with Composer!</td>
  </tr>
  <tr>
    <td><a href="https://mosaicml.com/jobs" target="_blank" rel="noopener noreferrer">We're Hiring!</a></td>
    <td>Join us! 🤩</td>
  </tr>
</tbody>
</table>

If you have any questions, please feel free to reach out to us on [Twiter](https://twitter.com/mosaicml), [email](mailto:community@mosaicml.com), or our [Community Slack](https://join.slack.com/t/mosaicml-community/shared_invite/zt-w0tiddn9-WGTlRpfjcO9J5jyrMub1dg)!

# 💫 Contributors
Composer is part of the broader Machine Learning community, and we welcome any contributions, pull requests, or issues!

# ✍️ Citation
```
@misc{mosaicml2022composer,
    author = {The Mosaic ML Team},
    title = {composer},
    year = {2021},
    howpublished = {\url{https://github.com/mosaicml/composer/}},
}
```
