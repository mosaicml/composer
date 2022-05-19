# 🧩 Stochastic Weight Averaging

![Untitled](https://storage.googleapis.com/docs.mosaicml.com/images/methods/swa.png)

The above image is from an extensive PyTorch blogpost about SWA:
[https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/)

Tags: `All`, `Increased Accuracy`, `Method`

## TL;DR

Stochastic Weight Averaging (SWA) maintains a running average of the weights towards the end of training. This leads to better generalization than conventional training.

## Attribution

["Averaging Weights Leads to Wider Optima and Better Generalization"](https://arxiv.org/abs/1803.05407) by Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Presented at the 2018 Conference on Uncertainty in Artificial Intelligence.

## Applicable Settings

Stochastic Weight Averaging is generally applicable across model architectures, tasks, and domains. It has been shown to improve performance in both [vision tasks](https://arxiv.org/abs/1803.05407) (e.g. ImageNet) as well as [NLP tasks](https://arxiv.org/abs/1902.02476).

## Hyperparameters

- `swa_start`: percent of training completed before stochastic weight averaging is applied. The default value is 0.8
- `swa_lr`: The final learning rate to anneal towards

## Example Effects

SWA leads to accuracy improvements of about 1-1.5% for ResNet50 on ImageNet. From the original paper and subsequent work by the authors (see their repo [here](https://github.com/izmailovpavel/torch_swa_examples)):


| Model      | Baseline SGD | SWA 5 epochs | SWA 10 epochs |
|------------|--------------|--------------|---------------|
| ResNet-50  | 76.15        | 76.83±0.01   | 76.97±0.05    |
| ResNet-152 | 78.31        | 78.82±0.01   | 78.94±0.07    |


Note that the implementation in the original papers is slightly different than the current [PyTorch implementation](https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/).

## Implementation Details

Our implementation is based off of the [PyTorch implementation](https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/), which treats SWA as an optimizer.  `SWALR` is imported from `torch.optim.swa_utils`. The current implementation first applies a cosine decay which reaches a fixed learning rate value, `swa_lr`, then begins maintaining a running average.

## Considerations

As per the paper, the majority of training should be completed (e.g. 75%-80%) before the final SWA learning rate is applied.

## Composability

Stochastic Weight Averaging composes well with other methods such as Mix Up and Label Smoothing on ImageNet.

There are studies in progress to see the effect of SWA on other image tasks and NLP.

---

### Code

```{eval-rst}
.. autoclass:: composer.algorithms.swa.SWA
    :members: match, apply
    :noindex:
```