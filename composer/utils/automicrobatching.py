import torch
import logging
from composer.core import State
from composer.utils import dist
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp._runtime_utils import _post_backward_final_callback
from collections import defaultdict
from packaging import version

if version.parse(torch.__version__) >= version.parse('2.3.0'):
    from torch.amp.grad_scaler import _refresh_per_optimizer_state  # type: ignore
else:
    from torch.cuda.amp.grad_scaler import _refresh_per_optimizer_state  # type: ignore

log = logging.getLogger(__name__)

__all__ = [
    '_fsdp_reshard_and_cleanup',
    '_closest_lower_power_of_2',
]

def _fsdp_reshard_and_cleanup(model: torch.nn.Module):
    """Manually reshard and clean up FSDP model.

    When an exception like OOM happens, _post_backward_final_callback, which
    is registered as a backward callback, will not run. We manually call it to cleanup
    loose memory.
    """
    for __, module in model.named_modules():
        if isinstance(module, FullyShardedDataParallel):
            if module.check_is_root():
                # Only call _post_backward_final_callback on root module. It will
                # traverse and reshard all FSDP sub-modules
                _post_backward_final_callback(module, module)

def _closest_lower_power_of_2(microbatch_size: int):
    """Find the highest lower power of 2 to serve as a lower bound device_train_microbatch_size when automicrobatching 
    searches downward, due to either thrashing or when a previously non-OOMing microbatch size is now OOMing.
    Args:
        microbatch_size (int): Current device train microbatch size.
    """
    if microbatch_size <= 1:
        return 1
    return 1 << ((microbatch_size - 1).bit_length() - 1)

