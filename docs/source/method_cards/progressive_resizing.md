# Progressive Image Resizing

![progressive_resizing_vision.png](https://storage.googleapis.com/docs.mosaicml.com/images/methods/progressive_resizing_vision.png)

Applicable Settings: `Vision`, `Increased GPU Throughput`, `Reduced GPU Memory Usage`, `Method`, `Curriculum`, `Speedup`

## TL;DR

Progressive Resizing works by initially shrinking the size of the training images, and slowly growing them back to their full size by the end of training. It reduces costs during the early phase of training, when the network may learn coarse-grained features that do not require details lost by reducing image resolution.

## Attribution

Inspired by the progressive resizing technique as [proposed by fast.ai](https://github.com/fastai/fastbook/blob/780b76bef3127ce5b64f8230fce60e915a7e0735/07_sizing_and_tta.ipynb).

## Applicable Settings

Progressive Resizing is intended for use on computer vision tasks where the network architecture can accommodate different sized inputs.

## Hyperparameters

- `initial_scale` - The initial scaling coefficient used to determine the height and width of images at the beginning of training. The default value of 0.5 converts a 224x224 image to a 112x112 image, for example.
- `finetune_fraction` - The fraction of training steps that should be devoted to training on the full-sized images. The default value of 0.2 means that there will be an initial training phase of 80% of `max_duration` whereby the input images are linearly scaled by a multiple from `initial_scale` to 1.0, followed by a fine-tuning phase of 20% of `max_duration` with a scale of 1.0.
- `mode` - The method by which images should be resized. Currently, the two implemented methods are `"crop"` , where the image is [randomly cropped](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomCrop) to the desired size, and `"resize"`, where the image is downsampled to the desired size using [bilinear interpolation](https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html).
- `resize_targets` - Whether the targets should be downsampled as well in the same fashion. This is appropriate for some tasks, such as segmentation, where elements of the output correspond to elements of the input image.

## Example Effects

When using Progressive Resizing, the early steps of training run faster than the later steps of training (which run at the original speed), since the smaller images reduce the amount of computation that the network must perform. Ideally, generalization performance is not impacted much by Progressive Resizing, but this depends on the specific dataset, network architecture, task, and hyperparameters. In our experience with ResNets on ImageNet, Progressive resizing improves training speed (as measured by wall clock time) with negligible effects on classification accuracy.

## Implementation Details

Our implementation of Progressive Resizing gives two options for resizing the images:

`mode = "crop"`  does a random crop of the input image to a smaller size. This mode is appropriate for datasets where scale is important. For example, we get better results using crops for ResNet-56 on CIFAR-10, where the objects are similar sizes to one another and the images are already low resolution.

`mode = "resize"` does downsampling with a bilinear interpolation of the image to a smaller size. This mode is appropriate for datasets where scale is variable, all the content of the image is needed each time it is seen, or the images are relatively higher resolution. For example, we get better results using resizing for ResNet-50 on ImageNet.

## Suggested Hyperparameters

`initial_scale = 0.5` is a reasonable starting point. This starts training on images where each side length has been reduced by 50%.

`finetune_fraction = 0.2` is a reasonable starting point for how long to train with full-sized images at the end of training. This reserves 20% of training at the end for training on full sized images.

## Considerations

Progressive Resizing requires that the network architecture be capable of handling different sized images. Additionally, since the early epochs of training require significantly less GPU compute than the later epochs, CPU/dataloading may become a bottleneck in the early epochs even if this isn't true in the late epochs.

Additionally, while we have not investigated this, Progressive Resizing may also change how sensitive the network is to different sizes of objects, or how biased the network is in favor of shape or texture.

## Composability

Progressive Resizing will interact with other methods that change the size of the inputs, such as Selective Backprop with downsampling and ColOut

## Detailed Results

Using the recommendations above, we ran a baseline ResNet-50 model on CIFAR-10 and ImageNet with and without progressive resizing. CIFAR-10 runs were done on a single NVIDIA 3080 GPU for 200 epochs. ImageNet runs were done on 8x NVIDIA 3080 GPUs for 90 epochs. Shown below are the validation set accuracies and time-to-train for each of these runs.



--------

## Code

```{eval-rst}
.. autoclass:: composer.algorithms.progressive_resizing.ProgressiveResizing
    :members: match, apply
    :noindex:
```

```{eval-rst}
.. autoclass:: composer.algorithms.progressive_resizing.resize_batch
    :noindex:
```
