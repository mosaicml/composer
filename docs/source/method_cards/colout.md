# 📢 ColOut

![col_out.png](https://storage.googleapis.com/docs.mosaicml.com/images/methods/col_out.png)

Several instances of an image of an apple taken from the CIFAR-100 dataset with ColOut augmentation.

Tags: `Vision`, `Increased GPU Throughput`, `Speedup`, `Method`, `Augmentation`, `Regularization`

## TL;DR

ColOut works by dropping a fraction of the rows and columns of an input image. If the fraction of rows/columns dropped isn't too large, the image content is not significantly altered, but the image size is reduced. The removal of rows and columns also introduces variability that can modestly degrade accuracy.

## Attribution

Cory Stephenson at MosaicML.

## Applicable Settings

ColOut is applicable to computer vision tasks where the network architecture is capable of handling different input image sizes.

## Hyperparameters

- `p_row` - The probability of dropping each row of the image.
- `p_col` - The probability of dropping each column of the image.
- `batch` - If set to True, the same rows and columns will be removed from every image in the batch, and operation will take place on the GPU. If set to False, rows and columns to be removed will be sampled independently for each image in the batch, and operation will take place on the CPU.

## Example Effects

ColOut reduces the size of images, reducing the number of operations per training step and consequently the total time to train the network. The variability induced by randomly dropping rows and columns can affect generalization performance. In our testing, we saw a decrease in accuracy (~.2%) in some models on ImageNet and a ~1% decrease in accuracy on CIFAR-10. This tradeoff of speed against accuracy is bargain we often find worth taking but should be carefully considered.

## Implementation Details

ColOut currently has two implementations, one which acts as an additional data augmentation for use in Pytorch dataloaders and another that operates on the batch level and runs on GPU. The optimal choice of which to use will depend on the amount of CPU and GPU compute available, as well as your tolerance for a small decrease in accuracy. If the workload is more CPU heavy, it may make sense to run ColOut batchwise on GPU, and vice versa. However, batchwise colout suffers a drop in validation accuracy compared to samplewise: .2% on CIFAR-10 and .1% on ImageNet.

## Suggested Hyperparameters

`p_row = 0.15` and `p_col = 0.15` strike a good balance between improving training throughput and limiting the negative impact on model accuracy. Setting `batch = True` also yields slightly lower accuracy, but for larger images this is offset by a large increase in throughput (~11% for ResNet-50 on ImageNet) because ColOut is only called once per batch and its operations are offloaded onto the GPU.

## Considerations

As mentioned above, the optimal way of invoking ColOut will depend on the specific hardware and workload. Additionally, some network architectures may not support running on smaller images than used by default.

## Composability

ColOut will show diminishing returns with other methods that change the size of images, such as Progressive Resizing and Selective Backdrop with downsampling. To the extent that ColOut serves as a form of regularization, combining regularization-based methods can lead to sublinear improvements in accuracy.

## Code

```{eval-rst}
.. autoclass:: composer.algorithms.colout.ColOut
    :members: match, apply
    :noindex:

.. autofunction:: composer.algorithms.colout.colout_image
    :noindex:
.. autofunction:: composer.algorithms.colout.colout_batch
    :noindex:
```

