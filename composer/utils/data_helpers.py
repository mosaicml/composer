from composer.core.data_spec import DataSpec
from torch.utils.data import DataLoader, Dataset
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Union, cast

if TYPE_CHECKING:
    from composer.core.evaluator import Evaluator

__all__ = ['_dataset_of']

def _dataset_of(self, dataloader: Optional[Union[Evaluator, DataSpec, DataLoader, Iterable]]) -> Optional[Dataset]:
    """Get the dataset contained by the given dataloader-like object.

    Args:
        dataloader (Evaluator | DataSpec | DataLoader | Iterable, optional): The dataloader, wrapped dataloader, or
            generic python iterable to get the dataset of, if applicable.

    Returns:
        Dataset: Its dataset, if there is one.
    """
    from composer.core.evaluator import Evaluator

    # If it's None, no dataset for you.
    if dataloader is None:
        return None

    # An Evaluator is a dataloader wrapped with metrics. Unwrap its dataloader.
    if isinstance(dataloader, Evaluator):
        dataloader = dataloader.dataloader

    # A DataSpec is a dataloader wrapped with an on-device transform. Unwrap its dataloader.
    if isinstance(dataloader, DataSpec):
        dataloader = dataloader.dataloader

    # If what we now have is an actual DataLoader, return its dataset. If not, return None.
    if isinstance(dataloader, DataLoader):
        return dataloader.dataset
    else:
        return None