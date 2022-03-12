# 🔢 Numerics

Composer supports several single and half-precision number formats including IEEE single-precision `fp32`, IEEE half-precision `fp16`, Google’s truncated `bf16` format and support for mixed precision training `amp`.  The following formats are supported per accelerator:

| Format | CPU | GPU - A100 | GPU - V100 | GPU - T4 |
| --- | --- | --- | --- | --- |
| bf16 | ➖ | ✅ | ➖ | ➖ |
| fp16 | ➖ | ✅ | ✅ | ✅ |
| fp32 (default) | ✅ | ✅ | ✅ | ✅ |
| amp | ➖ | ✅ | ✅ | ✅ |

When using the {class}`~.Trainer`, the number format can selected by specifying the `precision` argument during initialization. In the example below, we are training on a `gpu` device and using the `fp16` format:

```python
from composer import Trainer

# use mixed precision during training
trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  eval_dataloader=eval_dataloader,
                  max_duration='160ep',
                  device='gpu',
                  precision='amp')
```

```{note}
If the `precision` argument is not specified, the `Trainer` defaults to using `fp32`.
```

## Precision

When discussing number formats, the precision generally refers to the number of total bits used to represent decimal digits (e.g., `fp32` represents numbers using 32 bits of computer memory).  Higher precision formats can represent a greater range of numbers and more decimal places compared to lower precision formats.  However, since lower precision formats utilize fewer bits, they occupy a smaller memory footprint.  Additionally, depending on the accelerator, lower precision formats can typically unlock more compute resulting in greater overall throughput.  Therefore it is often advantageous to use half-precision formats to accelerate training.

For training, the most commonly used low precision formats are IEEE `fp16` and the Brain Floating Point `bf16` format.  While both formats utilize the same number of bits in memory, `bf16` offers a greater numerical range than `fp16` at the expense of being able to represent fewer decimal places, resulting in reduced sensitivity.

Please see the following references for additioanl details on the common number formats:

- `fp32`: [Single-precision Floating-point](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)
- `fp16`: [Half-precision Floating-point](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)
- `bf16`: [Brain Floating Point](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)

### Mixed Precision

Using half-precision number formats can boost model throughput, though not without issue.  The reduced representable range and sensitivity of these formats can cause several problems during training such as:

- Gradient underflow: Very small values can get zeroed out
- Incorrect weight updates: Small gradient values can lead to no weight updates
- Activation/Loss overflow: Very large values can roll over to representations of [Infinity](https://en.wikipedia.org/wiki/Infinity#Computing)(`Inf`)or [Not a Number](https://en.wikipedia.org/wiki/NaN) (`NaN`)

The result is reduced model accuracy.

The solution is to perform mixed precision training, where both single and half-precision formats are utilized strategically.  Mixed precision training is performed as follows:

1. Compute the forward pass and loss in half-precision, *except* for computations that can cause activations/loss overflow, which are performed in single-precision (e.g., BatchNorm)
2. Compute the backpropagation pass in half-precision
3. To increase numerical sensitivity, convert the model to single-precision and perform the optimizer step, enabling the weight update to be done more precisely
4. Convert the model back to half-precision

The procedure above mitigates the issues with overflow and imprecise weight updates. 

#### Gradient Scaling

In order to prevent gradient underflow, mixed precision training uses a technique called **Gradient Scaling** where the loss is multiplied by a scaling factor *before* backpropagation is performed (step 2 above). This trick scales the computed gradient values to a range where they cannot underflow.  Once the model has been converted to single-precision following the backward pass, the gradients are divided by the scaling factor to restore the gradient values to the original range before stepping the optimizer.

### AMP

Composer supports automatic mixed precision (AMP) training using PyTorch’s {mod}`torch.cuda.amp` package. The Composer {class}`~.Trainer` performs all the heavy lifting and tensor conversions automatically; the user simply has to set `precision='amp'` when initializing the {class}`~.Trainer`.

Another advantage of using Composer is support for optimizer closures when using mixed precision.  A closure is responsible for clearing the gradients and re-computing the loss, as required by some optimizers. The {class}`~.Trainer` supports closures when using `amp` if the optimizer object has the attribute {attr}`~.SAMOptimizer._step_supports_amp_closure` set to `True`.  Please see the [SAM Method Card](../method_cards/sam.md) for an implementation example on how closures are used.

```{warning}
AMP should not be used with DeepSpeed.  Please see [Deepspeed](distributed_training) for additional details.
```
