```{eval-rst}
:orphan:
```

# 🏋️‍♀️ Decoupled Weight Decay

Tags: `Best Practice`, `Increased Accuracy`, `Regularization`

## TL;DR

L2 regularization is typically considered equivalent to weight decay, but this equivalence only holds for certain optimizer implementations. Common optimizer implementations typically scale the weight decay by the learning rate, which complicates model tuning and hyperparameter sweeps by coupling learning rate and weight decay. Implementing weight decay explicitly and separately from L2 regularization allows for a new means of tuning regularization in models.

## Attribution

*[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)*, by Ilya Loshchilov and Frank Hutter. Published as a conference paper at ICLR 2019.

## Code and Hyperparameters

Unlike other methods, decoupled weight decay is not implemented as an algorithm, but instead provides two optimizers that can be used in place of existing common optimizers.

`DecoupledSGDW` optimizer (same as hyperparameters for `torch.optim.SGD`):

- `lr` - Learning rate.
- `momentum` - Momentum factor.
- `weight_decay` - Weight decay.
- `dampening` - Dampening for momentum.
- `nesterov` - Nesterov momentum.

`DecoupledAdamW` optimizer (same as hyperparameters for `torch.optim.Adam`)

- `lr` - Learning rate.
- `betas` - Coefficients used for computing running averages of gradient and its square.
- `eps` - Term for numerical stability.
- `weight_decay` - Weight decay.
- `amsgrad` - Use AMSGrad variant.

## Applicable Settings

Using decoupled weight decay is considered a best practice in most settings. `DecoupledSGDW` and `DecoupledAdamW` should always be used in place of their vanilla counterparts.

## Implementation Details

Unlike most of our other methods, we do not implement decoupled weight decay as an algorithm, instead providing optimizers that can be used as drop-in replacements for `torch.optim.SGD` and `torch.optim.Adam`, though note that some hyperparameter tuning may be required to realize full performance improvements.

The informed reader may note that Pytorch already provides a `torch.optim.AdamW` variant that implements Loshchilov et al.'s method. Unfortunately, this implementation has a fundamental bug owing to Pytorch's method of handling learning rate scheduling. In this implementation, learning rate schedulers attempt to schedule the weight decay (as Loschilov et al. suggest) by tying it to the learning rate. However, this means that weight decay is now implicitly tied to the initial learning rate, resulting in unexpected behavior where runs with different learning rates also have different effective weight decays. See [this line](https://github.com/pytorch/pytorch/blob/d921891f5788b37ea92eceddf7417d11e44290e6/torch/optim/_functional.py#L125).

## Considerations

There are no known negative side effects to using decoupled weight decay once it is properly tuned, as long as the original base optimizer is either `torch.optim.Adam` or `torch.optim.SGD`.

## Composability

Weight decay is a regularization technique, and thus is expected to yield diminishing returns when composed with other regularization techniques.
