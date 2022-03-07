# ⏮️ Selective Backprop

Tags: `Vision`, `NLP`, `Decreased Accuracy`, `Increased GPU Throughput`, `Method`, `Curriculum`, `Speedup`

## TL;DR

Selective Backprop prioritizes examples with high loss at each iteration, skipping examples with low loss. This speeds up training with limited impact on generalization.

## Attribution

*[Accelerating Deep Learning by Focusing on the Biggest Losers](https://arxiv.org/abs/1910.00762)* by Angela H. Jiang, Daniel L. K. Wong, Giulio Zhou, David G. Andersen, Jeffrey Dean, Gregory R. Ganger, Gauri Joshi, Michael Kaminsky, Michael Kozuch, Zachary C. Lipton, and Padmanabhan Pillai.

## Applicable Settings

Selective Backprop is broadly applicable across problems and modalities. It's one implementation among a class of methods that train first on the easy examples and focus on the hard examples later in training.

## Hyperparameters

- `start` - The fraction of training epochs elapsed at which to start pruning examples. For example, `start=0.5` with a total training epochs of 100 would have Selective Backprop begin at epoch 50.
- `end` -  The fraction of training epochs elapsed at which to stop pruning examples. This has to be larger than the value for `start`.
- `keep` - The fraction of examples in a batch that should be kept.
- `interrupt` - To alleviate some potential negative impacts on model performance, we do not prune the examples on a subset of batches within the interval that Selective Backprop is active. The `interrupt` parameter specifies the number of batches between these "vanilla", unpruned batches.
- `scale_factor` - The pruning process requires an additional forward pass in order to realize any speedup. Depending on the situation, this forward pass may be able to be performed on a downsampled version of the input. The `scale_factor` parameter controls the amount of downsampling to perform.

## Example Effects

Depending on the precise hyperparameters chosen, we see decreases in training time of around 10% without any degradation in performance. Larger values are possible but run into the speed-accuracy tradeoffs described below.

## Implementation Details

The goal of Selective Backprop is to reduce the number of examples the model sees to only those that still have high loss. This lets the model learn on fewer examples, speeding up forward and back propagation with limited impact on final model quality. To determine the per-example loss and which examples to skip, an additional, initial forward pass must be performed. These loss values are then used to weight a random sample of examples to use for training. For some data types, including images, it's possible to use a lower resolution version of the input for this additional forward pass. This minimizes the extra computation while maintaining a good estimate of which examples are difficult for the model.

## Suggested Hyperparameters

- `start`: Default: `0.5`. The default is a good value for most use cases. It lets the model train normally for most of the run while still providing a large boost in time-to-train.
- `end`: Default: `0.9`. The default is a good value for most use cases. It leaves a small amount of training at the end for fine-tuning on all examples in the dataset.
- `keep`: Default: `0.5`. We found a value of 0.5 to represent a good tradeoff that greatly improves speed at limited cost. This is likely to be the hyperparameter most worth tuning.
- `interrupt`: Default: `2`. We found that including some unpruned batches is worth the tradeoff in speed, though a value of `0` is worth considering.
- `scale_factor`: Default: `0.5`. If you are using a data type and model that can tolerate processing a downsampled input, this is definitely worthwhile. The default value of `0.5` yields good results with much less computation.

## Considerations

Selective Backprop has many tradeoffs between speed and accuracy that are readily apparent. The more data you eliminate for longer periods of training, the larger the potential impact in model performance. The default values we provide have worked well for us and strike a good balance.

## Composability

This method should be performed before data augmentation so that eliminated examples do not need to be augmented.

## Detailed Results

We have explored Selective Backprop primarily on image recognition tasks such as ImageNet and CIFAR-10. For both of these, we see large improvements in training time with little degradation in accuracy. The table below shows some examples using the default hyperparameters from above. For CIFAR-10, ResNet-56 was trained on 1x NVIDIA 3080 GPU for 160 epochs. For ImageNet, ResNet-50 was trained on 8x NVIDIA 3090 GPUs for 90 epochs.

```{eval-rst}
.. csv-table:: Selective Backprop
    :header: Dataset,Run,Validation Acc.,Time to Train

    ImageNet,Baseline,76.46%,5h 43m 8s
    ImageNet,+Selective Backprop,76.46%,5h 22m 14s
    CIFAR-10,Baseline,93.16%,35m 33s
    CIFAR-10,+Selective Backprop,93.32%,32m 36s
```

---

## Code

```{eval-rst}
.. autoclass:: composer.algorithms.selective_backprop.SelectiveBackprop
    :members: match, apply
    :noindex:

.. autofunction:: composer.algorithms.selective_backprop.select_using_loss
    :noindex:
```
