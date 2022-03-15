# 🧊 Stochastic Depth (Block)

AKA: Progressive Layer Dropping

![block_wise_stochastic_depth.png](https://storage.googleapis.com/docs.mosaicml.com/images/methods/block_wise_stochastic_depth.png)

From *[Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)* by Huang et al. 2016

Tags: `Method`,`NLP`, `Networks with Residual Connections`, `Vision`,`Regularization`, `Speedup`,`Decreased GPU Throughput`, `Decreased Wall Clock Time`, `Reduced GPU Memory Usage`

## TL;DR

Block-wise stochastic depth assigns every residual block a probability of dropping the transformation function, leaving only the skip connection, to regularize and reduce the amount of computation.

## Attribution

*[Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)* by Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Killian Weinberger. Published in ECCV in 2016.

*[Accelerating Training of Transformer-Based Language Models with Progressive Layer Dropping](https://arxiv.org/abs/2010.13369)* by Minjia Zhang and Yuxiong He. Published in NeurIPS 2020.

## Applicable Settings

Block-wise stochastic depth is only applicable to network architectures that include skip connections since they make it possible to drop parts of the network without disconnecting the network.

## Hyperparameters

- `stochastic_method` - Specifies the version of the stochastic depth method to use. Block-wise stochastic depth is specified by `stochastic_method=block`.
- `target_block_name` - The reference name for the module that will be replaced with a functionally equivalent stochastic block. For example, `target_block_name=ResNetBottleNeck` will replace modules in the model named `BottleNeck`.
- `drop_rate` - The base probability of dropping a block.
- `drop_distribution` - How the `drop_rate` is distributed across the model's blocks. The two possible values are "uniform" and "linear". "Uniform" assigns a single `drop_rate` across all blocks. "Linear" linearly increases the drop rate according to the block's depth, starting from 0 at the first block and ending with `drop_rate` at the last block.
- `use_same_gpu_seed` - Set to false to have the blocks to drop sampled independently on each GPU. Only has an effect when training on multiple GPUs.
- `drop_warmup` - Percentage of training to linearly warm-up the drop rate from 0 to `drop_rate`.

## Example Effects

For ResNet-50 on ImageNet, we used `drop_rate=0.2` and `drop_distribution=linear`. We measured a 5% decrease in training wall-clock time while maintaining a similar accuracy to the baseline. For ResNet-101 on ImageNet, we used `drop_rate=0.4` and `drop_distribution=linear`. We measured a 10% decrease in training wall-clock time while maintaining an accuracy within 0.1% of the baseline.

Huang et al. used `drop_rate=0.5` and `drop_distribution=linear` for ResNet-110 on CIFAR-10/100 and for ResNet-152 on ImageNet. They report that these hyperparameters result in a 25% reduction in training time with absolute accuracy differences of +1.2%, +2.8%, and -0.2% on CIFAR-10, CIFAR-100, and ImageNet respectively.

## Implementation Details

Every residual block in the model has an independent probability of skipping the transformation component by utilizing only the skip connection. This effectively drops the block from the computation graph for the current iteration. During inference, the transformation component is always used and the output is scaled by (1 - drop rate). This scaling compensates for skipping the transformation during training.

When using multiple GPUs to train, setting `use_same_gpu_seed` to false assigns each GPU a different random seed for sampling the blocks to drop. This effectively drops different proportions of the batch at each block, making block-wise stochastic depth more similar to example-wise stochastic depth. Typically, the result is increased accuracy and decreased throughput compared to dropping the same blocks across GPUs.

As proposed by Zhang et al., drop warmup specifies a portion of training to linearly increase the drop rate from 0 to `drop_rate`. This provides stability during the initial phase of training, improving convergence in some scenarios.

## Suggested Hyperparameters

The drop rate will primarily depend on the model depth. For ResNet50 on ImageNet, we find that a drop rate of 0.2 with a linear drop distribution has the largest reduction in training time while maintaining the same accuracy. We do not use any drop warmup and drop the same blocks across all GPUs.

- `drop_rate = 0.2`
- `drop_distribution = "linear"`
- `drop_warmup = 0.0dur`
- `use_same_gpu_seed = true`

## Considerations

Because block-wise stochastic depth reduces model capacity by probabilistically excluding blocks from training updates, the increased capacity of larger models allows them to accommodate higher block drop rates. For example, the largest drop rate that maintains accuracy on ResNet-101 is almost double the drop rate on ResNet-50. If using a model with only a few blocks, it is best to use a small drop rate or to avoid stochastic depth.

Although `use_same_gpu_seed = False` usually improves accuracy, there is a decrease in throughput that makes this setting undesirable in most scenarios. Only set this hyperparameter to false if the default settings are causing significant accuracy degradation.

## Composability

As a general rule, combining several regularization methods may have diminishing returns, and can even degrade accuracy. This may hold true when combining block-wise stochastic depth with other regularization methods.

Stochastic depth decreases forward and backward pass time, increasing GPU throughput. In doing so, it may cause data loading to become a bottleneck.


---

## Code

```{eval-rst}
.. autoclass:: composer.algorithms.stochastic_depth.StochasticDepth
    :members: match, apply
    :noindex:

.. autofunction:: composer.algorithms.stochastic_depth.apply_stochastic_depth
    :noindex:
```