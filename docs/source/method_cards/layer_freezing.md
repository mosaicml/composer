# Layer Freezing

Tags: `Vision`, `Decreased Accuracy`, `Increased GPU Throughput`, `Method`, `Backprop`, `Speedup`

## TL;DR

Layer Freezing gradually makes early modules not trainable ("freezing" them), saving the cost of backpropagating to and updating frozen modules.

## Attribution

Freezing layers is an old and common practice, but our precise freezing scheme most closely resembles:

- [Freezeout: Accelerate training by progressively freezing layers](https://arxiv.org/abs/1706.04983), by Brock et al. Posted to arXiv on 2017.

and the Freeze Training method of:

- [SVCCA: Singular vector canonical correlation analysis for deep learning dynamics and interpretability](https://arxiv.org/abs/1706.05806) by Raghu et al.. Presented at NIPS in 2017.

## Hyperparameters

- `freeze_start`: The fraction of epochs to run before freezing begins
- `freeze_level`: The fraction of the modules in the network to freeze by the end of training

## Applicable Settings

Layer freezing is in principle applicable to any model with many layers, but the MosaicML implementation currently only supports vision models.

## Example Effects

We've observed that layer freezing can increase throughput by ~5% for ResNet-50 on ImageNet, but decreases accuracy by 0.5-1%. This is not an especially good speed vs accuracy tradeoff. Existing papers have generally also not found effective tradeoffs on large-scale problems.

For ResNet-56 on CIFAR-100, we have observed an accuracy lift from 75.82% to 76.22% with a similar ~5% speed increase. However, these results used specific hyperparameters without replicates, and so should be interpreted with caution.

## Implementation Details

At the end of each epoch after `freeze_start`, the algorithm traverses the ownership tree of `torch.nn.Module` objects within one's model in depth-first order to obtain a list of all modules. Note that this ordering may differ from the order in which modules are actually used in the forward pass.

Given this list of modules, the algorithm computes how many modules to freeze. This number increases linearly over time such that no modules are frozen at `freeze_start` and a fraction equal to `freeze_level` are frozen at the end of training.

Modules are frozen by removing their parameters from the optimizer's `param_groups`. However, their associated state `dict` entries are not removed.

## Suggested Hyperparameters

- `freeze_start` should be at least 0.1 to allow the network a warmup period.

## Considerations

We have yet to observe a significant improvement in the tradeoff between speed and accuracy using this method. However, there may remain other tasks for which the technique works well. Moreover, freezing layers can be useful for [understanding a network](https://arxiv.org/abs/1706.05806).

## Composability

Layer freezing is a relaxed version of [early stopping](https://en.wikipedia.org/wiki/Early_stopping) that stops training the model gradually, rather than all at once. It can therefore be understood as a form of regularization. Combining multiple regularization methods often yields diminishing improvements to accuracy.

---

## Code

```{eval-rst}
.. autoclass:: composer.algorithms.layer_freezing.LayerFreezing
    :members: match, apply
    :noindex:

.. autofunction:: composer.algorithms.layer_freezing.freeze_layers
  :noindex:
```