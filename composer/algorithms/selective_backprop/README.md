# ⏮️ Selective Backprop

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

`Computer Vision`

Selective Backprop prioritizes examples with high loss at each iteration, skipping backpropagation on examples with low loss.
This speeds up training with limited impact on generalization.

| ![SelectiveBackprop](https://storage.googleapis.com/docs.mosaicml.com/images/methods/selective-backprop.png) |
|:--|
|*Four examples are forward propagated through the network. Selective backprop only backpropagates the two examples that have the highest loss.*|

<!--## How to Use

### Functional Interface

TODO(ABHI): Fix and comments here describing what happens below.


```python
import composer.functional as cf

def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())
    loss_fn = F.cross_entropy
    model.train()

    for epoch in range(num_epochs):
        for X, y in train_loader:
            y_hat = model(X)
            loss = loss_fn(y_hat, smoothed_targets)
            loss.backward()
            opt.step()
            opt.zero_grad()
```

### Composer Trainer

TODO(Abhi): Fix and add comments here describing what happens below.

```python
from composer.algorithms import LabelSmoothing
from composer.trainer import Trainer

trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  max_duration='1ep',
                  algorithms=[])

trainer.fit()
```

### Implementation Details

TODO(ABHI): Briefly describe what happens under the hood here.-->

## Suggested Hyperparameters

We recommend setting `start=0.5` and `end=0.9`, which starts performing Selective Backprop halfway through training and stops performing Selective Backprop 90% of the way through training.
We found that the network performs better when it has time at both the beginning and end of training where it is exposed to all training examples.

We recommend setting `keep=0.5`, which keeps half of the examples on each step where Selective Backprop is performed.
We found a value of 0.5 to represent a good tradeoff that greatly improves speed at limited cost to quality. This is likely to be the hyperparameter most worth tuning.

We recommend setting `interrupt=2`, which performs a standard training batch after every two batches of Selective Backprop.
We found that including some unmodified batches is worth the tradeoff in speed, although a value of 0 is also worth considering.

We recommend setting `scale_factor=0.5`, which downsamples the height and width image examples by 50% on the first forward pass (the one that selects which examples to train on). This mitigates the cost of that additional forward pass.

> ❗ The Network Must Be Able to Handle Lower Resolution Images to Use `scale_factor`
> 
> Using the `scale_factor` hyperparameter require a network and data preparation pipeline capable of handling lower resolution images. If your pipeline and network are not capable of doing so, set this hyperparameter to 1.0.
> 
## Technical Details

The goal of Selective Backprop is to reduce the number of examples the model sees to only those that still have high loss.
This lets the model learn on fewer examples, speeding up forward and back propagation with limited impact on final model quality.
To determine the per-example loss and which examples to skip, an additional, initial forward pass must be performed.
These loss values are then used to weight the examples, and the network is trained on a sample of examples selected based on those weights.

> 🚧 Requires an Additional Forward Pass on Each Step
> 
> Selective backprop must perform two forward passes on each training step. The first forward pass computes the loss for each example. The main training step then occurs, with a forward and backward pass for any examples selected after the first forward pass.
> This additional forward pass can slow down training depending on the number of examples that are dropped.
> The forward pass accounts for approximately one third of the cost of each training step, so at least a third of the examples must hypothetically be dropped for selective backprop to improve throughput.

> ✅ The Cost of the Additional Forward Pass Can Be Mitigated
> 
> For some data types, including images, it is possible to mitigate the cost of this additional forward pass.
> The first forward pass does not need to be as precise as the second forward pass, since it is only selecting how to weight the examples, not how to update the network.
> As such, this first forward pass can be approximate.
> Our implementation of Selective Backprop for image datasets provides the option to perform the first forward pass at lower resolution (see the `scale_factor` hyperparameter), reducing the burden imposed by this additional forward pass.

Depending on the precise hyperparameters chosen, we see decreases in training time of around 10% without any degradation in performance. Larger values are possible but run into speed-accuracy tradeoffs.
Namely, the more data that is eliminated for longer periods of training, the larger the potential impact in model performance.
The default hyperparameters listed above have worked well for us and strike a good balance between speedup and maintaining model quality.
We found several techniques for mitigating accuracy degradation, including starting Selective Backprop mid-way through training (see the `start` hyperparameter), disabling it before the end of training to allow fine-tuning with the standard training regime (see the `end` hyperparameter), and mixing in occasional iterations where all data is used (see the `interrupt` hyperparameter).

We have explored Selective Backprop primarily on image recognition tasks such as ImageNet and CIFAR-10. For both of these, we see large improvements in training time with little degradation in accuracy. The table below shows some examples using the default hyperparameters from above. For CIFAR-10, ResNet-56 was trained on 1x NVIDIA 3080 GPU for 160 epochs. For ImageNet, ResNet-50 was trained on 8x NVIDIA 3090 GPUs for 90 epochs.

| Dataset | Run | Validation Accuracy | Time to Train |
|---------|-----|---------------------|---------------|
| ImageNet | Baseline | 76.46% | 5h 43m 8s|
|  | + Selective Backprop | 76.46% | 5h 22m 14s|
| CIFAR-10 | Baseline | 93.16% | 35m 33s |
|  | + Selective Backprop | 93.32% | 32m 36s|

> ✅ Selective Backprop Improves the Tradeoff Between Quality and Training Speed
>
>In our experiments, Selective Backprop improves the attainable tradeoffs between training speed and the final quality of the trained model. In some cases, it leads to slightly lower quality than the original model for the same number of training steps. However, Selective Backprop increases training speed so much (via improved throughput) that it is possible to train for more steps, recover accuracy, and still complete training in less time.

## Attribution

[*Accelerating Deep Learning by Focusing on the Biggest Losers*](https://arxiv.org/abs/1910.00762) by Angela H. Jiang, Daniel L. K. Wong, Giulio Zhou, David G. Andersen, Jeffrey Dean, Gregory R. Ganger, Gauri Joshi, Michael Kaminsky, Michael Kozuch, Zachary C. Lipton, and Padmanabhan Pillai. Released on arXiv in 2019.

*The Composer implementation of this method and the accompanying documentation were produced by Abhi Venigalla at MosaicML.*
