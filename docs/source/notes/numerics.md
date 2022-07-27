# 🔢 Numerics

Composer supports several single and half-precision number formats including IEEE single-precision `fp32`, IEEE half-precision
`fp16`, Google’s truncated `bf16` format, and mixed precision training `amp`.  The following formats are supported per
accelerator:

| Format         | CPU | GPU - A100 | GPU - V100 | GPU - T4 |
| -------------- | --- |----------- | ---------- | -------- |
| bf16           | ➖   | ✅         | ➖         | ➖        |
| fp16           | ➖   | ✅         | ✅         | ✅        |
| fp32 (default) | ✅   | ✅         | ✅         | ✅        |
| amp            | ➖   | ✅         | ✅         | ✅        |

When using the {class}`~.Trainer`, the number format can be selected by specifying the `precision` argument during
initialization. In the example below, we are training on a `gpu` device using [Automatic Mixed Precision](amp) (`amp`):

<!--pytest.mark.gpu-->
<!--
```python
from torch.utils.data import DataLoader
from tests.common import RandomClassificationDataset, SimpleModel

model = SimpleModel()
train_dataloader = DataLoader(RandomClassificationDataset())
eval_dataloader = DataLoader(RandomClassificationDataset())
```
-->
<!--pytest-codeblocks:cont-->
```python
from composer import Trainer

# use mixed precision during training
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration='160ep',
    device='gpu',
    precision='amp'
)
```

```{note}
If the `precision` argument is not specified, the `Trainer` defaults to using `fp32`.
```

## Precision

When discussing number formats, the precision generally refers to the number of bits used to represent decimal digits
(e.g. [IEEE Single-precision Floating-point (`fp32`)](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)
represents numbers using 32-bits of computer memory).  Higher precision formats can represent a greater range of numbers and
more decimal places compared to lower precision formats.  However, lower precision formats require less memory and, on many
accelerators, enable greater compute throughput.  Therefore, it is often advantageous to use lower precision formats to
accelerate training.

For training, the most commonly used low precision formats are the [IEEE Half-precision Floating-point (`fp16`)](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) and the [Brain Floating-point (`bf16`)](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) formats. While both formats utilize the same number of bits in memory, `bf16` offers a greater numerical range than `fp16` at the expense of being able to represent fewer decimal places.

(amp)=
## Automatic Mixed Precision (AMP) Training

Using half-precision number formats can boost model throughput, though not necessarily for free.
The reduced representable range and increased rounding error of these formats can cause several problems during training, such as:

(gradient_underflow)=
- Gradient underflow: Very small values can get zeroed out
- Incorrect weight updates: Small gradient values can lead to no weight updates
- Activation/Loss overflow: Very large values can roll over to representations of [Infinity](https://en.wikipedia.org/wiki/Infinity#Computing)(`Inf`) or [Not a Number](https://en.wikipedia.org/wiki/NaN) (`NaN`)

The result can be reduced model accuracy or training divergence.

The solution is to perform mixed precision training, where both single (`fp32`) and half-precision (`fp16`) formats are utilized strategically to avoid the issues above. Composer supports Automatic Mixed Precision (AMP) training using PyTorch’s {mod}`torch.cuda.amp` package. The Composer {class}`~.Trainer` performs all the heavy lifting and tensor conversions automatically; the user simply has to set `precision='amp'` when initializing the {class}`~.Trainer`.

Mixed precision training is usually performed as follows:

1. Compute the forward pass and loss in half-precision, *except* for computations that can cause activations/loss to overflow.
These are performed in single-precision (e.g., BatchNorm).
2. Perform the backward pass in half-precision.
3. Store the weights and perform the optimizer step in single precision, enabling the weight update to be done more precisely.
4. Convert the model back to half-precision.

The procedure above mitigates the issues with overflow and imprecise weight updates.

```{warning}
AMP should not be used with DeepSpeed.  Please see [Deepspeed](distributed_training) for additional details.
```

### Gradient Scaling

In order to prevent the [gradient underflow](gradient_underflow) issue, mixed precision training uses a technique called
**Gradient Scaling** where the loss is multiplied by a scaling factor *before* backpropagation is performed (step 2 above).
This trick scales the computed gradient values to a range where they cannot underflow. Once the model has been converted to
single-precision following the backward pass, the gradients are divided by the scaling factor to restore the gradient values to
the original range before stepping the optimizer. The scaling factor is automatically tuned such that when an underflow
exception is caught, the violating batch is discarded and the scaling factor is adjusted for the next batch to prevent the
underflow scenario; thus, no manual user intervention is required.

### Closures

Another advantage of using Composer is support for optimizer closures when using mixed precision. A closure is responsible for
clearing the gradients and re-computing the loss, as required by some optimizers. The {class}`~.Trainer` supports closures when
using `amp` if the optimizer object has the attribute `_step_supports_amp_closure` set to `True`.
Please see the [SAM Method Card](../method_cards/sam.md) for an example on how closures are used and the
{class}`~.SAMOptimizer` implementation for an example on how to use the `_step_supports_amp_closure` flag.
