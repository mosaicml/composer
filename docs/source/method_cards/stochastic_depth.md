# 🧊 Stochastic Depth (Block)

AKA: Progressive Layer Dropping

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution) - [\[API Reference\]](#api-reference)

Block-wise stochastic depth assigns every residual block a probability of dropping the transformation function, leaving only the skip connection, to regularize and reduce the amount of computation.

![block_wise_stochastic_depth.png](https://storage.googleapis.com/docs.mosaicml.com/images/methods/block_wise_stochastic_depth.png)

From *[Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)* by Huang et al. 2016

## Suggested Hyperparameters

The drop rate will primarily depend on the model depth. For ResNet50 on ImageNet, we find that a drop rate of 0.2 with a linear drop distribution has the largest reduction in training time while maintaining the same accuracy. We do not use any drop warmup and drop the same blocks across all GPUs.

- `drop_rate = 0.2`
- `drop_distribution = "linear"`
- `drop_warmup = 0.0dur`
- `use_same_gpu_seed = True`

## Technical Details

> 🚧 Block-wise stochastic depth is only applicable to network architectures that include skip connections since they make it possible to drop parts of the network without disconnecting the network.

Every residual block in the model has an independent probability of skipping the transformation component by utilizing only the skip connection. This effectively drops the block from the computation graph for the current iteration. During inference, the transformation component is always used and the output is scaled by (1 - drop rate). This scaling compensates for skipping the transformation during training.

> 🚧 Stochastic depth decreases forward and backward pass time, increasing GPU throughput. In doing so, it may cause data loading to become a bottleneck.

When using multiple GPUs to train, setting `use_same_gpu_seed=False` assigns each GPU a different random seed for sampling the blocks to drop. This effectively drops different proportions of the batch at each block, making block-wise stochastic depth  similar to example-wise stochastic depth. Typically, the result is increased accuracy and decreased throughput compared to dropping the same blocks across GPUs.

As proposed by Zhang et al., drop warmup specifies a portion of training to linearly increase the drop rate from 0 to `drop_rate`. This provides stability during the initial phase of training, improving convergence in some scenarios.


> ✅ Blockwise Stochastic Depth Improves the Tradeoff Between Quality and Training Speed
>
> In our experiments, Blockwise Stochastic Depth improves the attainable tradeoffs between training speed and the final quality of the trained model.
> We recommend Label Smoothing for image classification tasks.

For ResNet-50 on ImageNet, we used `drop_rate=0.2` and `drop_distribution=linear`. We measured a 5% decrease in training wall-clock time while maintaining a similar accuracy to the baseline. For ResNet-101 on ImageNet, we used `drop_rate=0.4` and `drop_distribution=linear`. We measured a 10% decrease in training wall-clock time while maintaining an accuracy within 0.1% of the baseline.

Huang et al. used `drop_rate=0.5` and `drop_distribution=linear` for ResNet-110 on CIFAR-10/100 and for ResNet-152 on ImageNet. They report that these hyperparameters result in a 25% reduction in training time with absolute accuracy differences of +1.2%, +2.8%, and -0.2% on CIFAR-10, CIFAR-100, and ImageNet, respectively.

Because block-wise stochastic depth reduces model capacity by probabilistically excluding blocks from training updates, the increased capacity of larger models allows them to accommodate higher block drop rates. For example, the largest drop rate that maintains accuracy on ResNet-101 is almost double the drop rate on ResNet-50. If using a model with only a few blocks, it is best to use a small drop rate or to avoid stochastic depth.

> 🚧 Composing Regularization Methods
>
> As general rule, composing regularization methods may lead to diminishing returns in
> quality improvements. This may hold true when combining block-wise stochastic depth with other regularization methods.

## Attribution

*[Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)* by Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Killian Weinberger. Published in ECCV in 2016.

*[Accelerating Training of Transformer-Based Language Models with Progressive Layer Dropping](https://arxiv.org/abs/2010.13369)* by Minjia Zhang and Yuxiong He. Published in NeurIPS 2020.

## API Reference

See {mod}`composer.algorithms.stochastic_depth`
