# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A custom safer checkpoint format that does not allow for arbitrary code execution (unlike pickle)."""

import json
import shutil
import tempfile
import zipfile
from datetime import timedelta
from pathlib import Path
from types import NoneType
from typing import Any

import numpy as np
import numpy.typing as npt
import safetensors.torch
import torch

import composer.utils.checkpoint
from composer.loggers import LoggerDestination
from composer.utils.object_store import ObjectStore

_original_write_checkpoint_file = composer.utils.checkpoint._write_checkpoint_file
_original_download_checkpoint = composer.utils.checkpoint.download_checkpoint
_original_safe_torch_load = composer.utils.checkpoint.safe_torch_load


def _patched_write_checkpoint_file(state_dict: dict[str, Any], filename: str) -> None:
    if Path(filename).suffix == '.safe':
        SafeState.from_data(state_dict).save(Path(filename))
    else:
        _original_write_checkpoint_file(state_dict, filename)


def _patched_download_checkpoint(
    path: str,
    node_checkpoint_folder: str,
    object_store: ObjectStore | LoggerDestination | None,
    progress_bar: bool,
    fsdp_sharded_state_dict_enabled: bool = False,
    deepspeed_sharded_checkpoint: bool = False,
) -> tuple[str, str | None, bool]:
    """A monkeypatched version of _download_checkpoint to enable the safe checkpoint format.

    In the case of a local checkpoint file that is not a .tar archive, `_download_checkpoint` writes a single symlink
    into a temporary directory and later on `safe_torch_load` is called on that file. This patch sets the file extension
    of the symlink so that the patched `safe_torch_load` can identify it as being '.safe' instead of '.pt'.
    """
    composer_states_filepath, extracted_checkpoint_folder, extracted_rank_n = _original_download_checkpoint(
        path,
        node_checkpoint_folder,
        object_store,
        progress_bar,
        fsdp_sharded_state_dict_enabled,
        deepspeed_sharded_checkpoint,
    )
    if Path(path).suffix == '.safe':
        symlink_path = Path(composer_states_filepath)
        composer_states_filepath = str(symlink_path.rename(symlink_path.with_suffix('.safe')))
    return composer_states_filepath, extracted_checkpoint_folder, extracted_rank_n


def _patched_safe_torch_load(
    composer_states_filepath: Path | str,
    map_location: str = 'cpu',
    load_fsdp_monolith_rank0_only: bool = False,
) -> dict[str, Any]:
    if Path(composer_states_filepath).suffix == '.safe':
        return SafeState.load(Path(composer_states_filepath)).to_data()  # type: ignore[no-any-return]
    else:
        return _original_safe_torch_load(composer_states_filepath, map_location, load_fsdp_monolith_rank0_only)


def install_safe_checkpoint() -> None:
    """Patch composer to enable the safer checkpoint format when saving with `.safe`."""
    composer.utils.checkpoint._write_checkpoint_file = _patched_write_checkpoint_file
    composer.utils.checkpoint.download_checkpoint = _patched_download_checkpoint
    composer.utils.checkpoint.safe_torch_load = _patched_safe_torch_load


class SafeState:
    """A data structure that can be serialized and deserialized without executing arbitrary code (unlike Pickle)."""

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        ndarrays: dict[str, npt.NDArray[Any]],
        other_data: Any,
    ) -> None:
        self._tensors = tensors
        self._ndarrays = ndarrays
        self._other_data = other_data

    @staticmethod
    def load(path: Path) -> 'SafeState':
        if not path.exists():
            raise FileNotFoundError(path)
        with tempfile.TemporaryDirectory(prefix='SafeState') as tmpdir:
            output_dir = Path(tmpdir)
            shutil.unpack_archive(path, extract_dir=output_dir, format='gztar')
            tensors = safetensors.torch.load_file(output_dir / 'tensors.safetensors')
            ndarrays = np.load(output_dir / 'ndarrays.npz', allow_pickle=False)
            with (output_dir / 'other.json').open() as f:
                other_data = json.load(f)
        return SafeState(tensors, ndarrays, other_data)

    def save(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(path)
        with tempfile.TemporaryDirectory(prefix='SafeState') as tmpdir:
            tmp_path = Path(tmpdir)
            safetensors.torch.save_file(self._tensors, tmp_path / 'tensors.safetensors')
            _savez(tmp_path / 'ndarrays.npz', self._ndarrays)
            with (tmp_path / 'other.json').open('w') as f:
                json.dump(self._other_data, f)
            filename = shutil.make_archive(str(path), format='gztar', root_dir=tmp_path)
            Path(filename).rename(path)

    @staticmethod
    def from_data(data: Any) -> 'SafeState':
        objects = _NonJsonObjects({}, {})
        extracted_data = _extract_objects(data, objects)
        return SafeState(objects.tensors, objects.ndarrays, extracted_data)

    def to_data(self) -> Any:
        objects = _NonJsonObjects(self._tensors, self._ndarrays)
        return _insert_objects(self._other_data, objects)


def _savez(output_path: Path, ndarrays: dict[str, npt.NDArray[Any]]) -> None:
    """Save numpy arrays to a `.npz` file.

    Based on np.lib.npyio._savez() which is used by np.savez() but with allow_pickle=False.
    """
    zipf = zipfile.ZipFile(output_path, mode='w', compression=zipfile.ZIP_STORED, allowZip64=True)
    for key, val in ndarrays.items():
        with zipf.open(key + '.npy', 'w', force_zip64=True) as fid:
            np.lib.format.write_array(fid, np.asanyarray(val), allow_pickle=False, pickle_kwargs={})
    zipf.close()


_PRIMITIVE_TYPES = (float, int, str, bool, NoneType)
_KEY = '__safe_storage_obj'


class _NonJsonObjects:

    def __init__(self, tensors: dict[str, torch.Tensor], ndarrays: dict[str, npt.NDArray[Any]]) -> None:
        self.tensors = tensors
        self.ndarrays = ndarrays
        self._id_counter = 0

    @property
    def next_id(self) -> str:
        next_id = str(self._id_counter)
        self._id_counter += 1
        return next_id

    def add_tensor(self, item: torch.Tensor) -> str:
        item_id = self.next_id
        self.tensors[item_id] = item
        return item_id

    def add_ndarray(self, item: npt.NDArray[Any]) -> str:
        item_id = self.next_id
        self.ndarrays[item_id] = item
        return item_id


def _extract_objects(data: Any, objects: _NonJsonObjects) -> Any:
    if isinstance(data, _PRIMITIVE_TYPES):
        return data
    elif isinstance(data, list):
        return [_extract_objects(item, objects) for item in data]
    elif isinstance(data, dict):
        return {k: _extract_objects(v, objects) for k, v in data.items()}
    elif isinstance(data, tuple):
        return {_KEY: 'tuple', 'items': [_extract_objects(item, objects) for item in data]}
    elif isinstance(data, timedelta):
        return {_KEY: 'timedelta', 'seconds': data.total_seconds()}
    elif isinstance(data, torch.Tensor):
        return {_KEY: 'tensor', 'id': objects.add_tensor(data)}
    elif isinstance(data, np.ndarray):
        return {_KEY: 'ndarray', 'id': objects.add_ndarray(data)}
    else:
        raise TypeError(f'unsupported type: {type(data).__name__}')


def _insert_objects(data: Any, objects: _NonJsonObjects) -> Any:
    if isinstance(data, _PRIMITIVE_TYPES):
        return data
    elif isinstance(data, list):
        return [_insert_objects(item, objects) for item in data]
    elif isinstance(data, dict):
        if _KEY in data:
            key = data[_KEY]
            if key == 'tensor':
                return objects.tensors[data['id']]
            elif key == 'ndarray':
                return objects.ndarrays[data['id']]
            elif key == 'tuple':
                return tuple(_insert_objects(v, objects) for v in data['items'])
            elif key == 'timedelta':
                return timedelta(seconds=data['seconds'])
            else:
                raise ValueError(f'unknown key: "{key}"')
        else:
            return {k: _insert_objects(v, objects) for k, v in data.items()}
    else:
        raise TypeError(f'unsupported type: {type(data).__name__}')
