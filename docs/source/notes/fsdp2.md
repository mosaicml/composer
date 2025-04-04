# FSDP2 Migration Guide

## Intro

We want to update our DLE codebase (Composer, LLM Foundry, etc) to use Pytorch FSDP2. The main differences between FSDP1 and 2 are:

- FSDP1 used a flat 1D tensor buffer underneath which complicates model parameters while FSDP2 recovers per tensor semantics with DTensor
- FSDP runtime like prefetch/reshard is managed by the FSDP wrapper module/class in FSDP1 while they are handled by forward/backward hooks registered on original model in FSDP2
- A more detailed comparison between FSDP1 vs FSDP2 and supported arguments can be found [here](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md).

## The main benefits of FSDP2 are:

- **DTensor based**: It is the modern and preferred abstraction of torch's distributed approach
- **Integration with parallelisms**: Marrying FSDP and various parallelisms like TP/SP/PP, etc and Dynamo (`torch.compile`)
- **Better parameter representation**: It provides a sharding representation of each Parameter in the model and relevant state_dict, serving as an interface to reshard between different parallelism plans and frameworks, as adopted in RL frameworks like VeRL
- **Simpler APIs**: This makes the codebase more readable and maintainable
- **Potential slight performance gain**

> **NOTE:**
> The design philosophy of FSDP2 and Torchtitan (or torch's modern distributed approach) is to make parallelisms declarative on top of model definition (similar to JAX) instead of directly inlined with model definition (like Megatron/DeepSpeed). This is generally more scalable and modularized and we adopted the same philosophy by isolating model definition and parallelisms through (FSDP/TP) YAML configs.

## Functionality between FSDP2 vs FSDP1 wrapper in Composer

### FSDP2 wrapper `prepare_fully_shard` is backward compatible with FSDP1 wrapper `prepare_fsdp_module` on following supports:

#### Inherited from FSDP1 to FSDP2
| Feature | Details |
|---------|---------|
| **Mixed Precision** | from MixedPrecision to MixedPrecisionPolicy |
| **CPU offload** | from CPUOffload to OffloadPolicy |

#### Integrated

| Feature | Details |
|---------|---------|
| **Auto submodule level wrapping** | `prepare_fsdp_module` used sub-module level wrapping (direct children), which is in line with FSDP2's per submodule design, so we will support wrap `fully_shard` on each sub-module. |
| **State Checkpointing** | FSDP1 with explicit `device_mesh` produces DTensor based statedict. In FSDP2 we support all ckpts saved with this format. Checkpoints saved with ShardedTensor (FSDP1 + SHARDED_STATE_DICT) won't be supported. |
| **Weight Tying** | FSDP2 leaves it to users. We need to keep it safe as we did in FSDP1 `prepare_fsdp_module`. In principle, if two submodules share parameters, FSDP2 `fully_shard` would be applied to their parent module instead of on them. |
| **Microbatching** | Continue support w/ per microbatch AG/RS and prevents deadlock due to collective communication. |
| **HSDP** | Is already supported by FSDP1 through `device_mesh`. Should be a smooth transition to FSDP2. |

#### Potentially decouplable

| Feature | Details |
|---------|---------|
| **Optimizers** | FSDP2 suggests Optimizer construction post FSDP, while Optimizer is Optional in Composer. We could support this optional semantics and it is up to the user to construct optimizers pre/post FSDP. |
| **Activation checkpointing** | Continue support with per direct children submodule granularity, but in a standalone function rather than the FSDP2 wrapper |

### New functionality

| Feature | Details |
|---------|---------|
| **HSDP with flexible reshard degree** | Int reshard_after_forward |
| **2D Composability with TP/SP/EP** | - Legalization of order of applying TP/EP + FSDP<br>- Legalization of order of applying SP + FSDP (possible hook/stream order) |
