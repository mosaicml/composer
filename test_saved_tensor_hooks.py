from contextlib import contextmanager
import argparse
import torch
import torch.nn as nn

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing, offload_wrapper


@contextmanager
def check_saved_tensor_device(nested_hooks: bool = False):
    """Context manager to check the device of tensors saved by activation checkpointing."""

    _captured_devices = []

    # This is used whenever a tensor is saved for backward
    # We use this hook to capture the device of a tensor when it is saved
    def pack_hook(saved_tensor):
        print("pack hook")
        _captured_devices.append(saved_tensor.device.type)
        return saved_tensor

    # This is used whenever a saved tensor is used for backward
    def unpack_hook(packed_tensor):
        print("unpack hook")
        return packed_tensor

    def _inner_pack_hook(saved_tensor):
        print("inner pack hook")
        return saved_tensor

    def _inner_unpack_hook(packed_tensor):
        print("inner unpack hook")
        return packed_tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        if nested_hooks:
            with torch.autograd.graph.saved_tensors_hooks(_inner_pack_hook, _inner_unpack_hook):
                yield _captured_devices
        else:
            yield _captured_devices


def test_saved_tensor_device(checkpoint_module: bool = False, nested_hooks: bool = False):
    """Test that the device of tensors saved by activation checkpointing is correct."""
    linear = nn.Linear(4, 4)
    sequential = nn.Sequential(linear)
    with check_saved_tensor_device(nested_hooks) as captured_devices:
        a = torch.randn(2, 4)
        if checkpoint_module:
            apply_activation_checkpointing(sequential, offload_wrapper)
        b = sequential(a)
    if not checkpoint_module:
        if nested_hooks:
            # override pack_hook
            assert captured_devices == [], f"Captured devices: {captured_devices}"
        else:
            assert captured_devices == ['cpu'], f"Captured devices: {captured_devices}"
    else:
        # override pack_hook and _inner_pack_hook
        assert captured_devices == [], f"Captured devices: {captured_devices}"


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--checkpoint-module", action="store_true")   
    args.add_argument("--nested-hooks", action="store_true")
    args = args.parse_args()
    test_saved_tensor_device(args.checkpoint_module, args.nested_hooks)
