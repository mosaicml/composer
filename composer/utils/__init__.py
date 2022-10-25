# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helper utilities."""
import warnings

from composer.utils.batch_helpers import batch_get, batch_set
from composer.utils.checkpoint import load_checkpoint, save_checkpoint
from composer.utils.collect_env import configure_excepthook, disable_env_report, enable_env_report, print_env
from composer.utils.device import get_device, is_tpu_installed
from composer.utils.file_helpers import (create_symlink_file, ensure_folder_has_no_conflicting_files,
                                         ensure_folder_is_empty, format_name_with_dist, format_name_with_dist_and_time,
                                         get_file, is_tar)
from composer.utils.import_helpers import MissingConditionalImportError, import_object
from composer.utils.inference import export_for_inference, export_with_logger, quantize_dynamic
from composer.utils.iter_helpers import IteratorFileStream, ensure_tuple, map_collection
from composer.utils.misc import is_model_deepspeed, is_model_fsdp, is_notebook, model_eval_mode
from composer.utils.object_store import (LibcloudObjectStore, ObjectStore, ObjectStoreTransientError, S3ObjectStore,
                                         SFTPObjectStore)
from composer.utils.retrying import retry
from composer.utils.string_enum import StringEnum


def warn_yahp_deprecation() -> None:
    warnings.warn(
        'yahp-based workflows are deprecated and will be removed in a future release. Please'
        'migrate to using other configuration managers and create the Trainer objects directly.'
        'v0.10 will be the last release to support yahp.',
        DeprecationWarning,
        stacklevel=2)


def warn_streaming_dataset_deprecation(old_version: int, new_version: int) -> None:
    warnings.warn(
        f'streaming dataset version {old_version} is deprecated and will be removed in the future. '
        f'Please migrate to using streaming dataset version {new_version}',
        DeprecationWarning,
        stacklevel=2)


__all__ = [
    'ensure_tuple',
    'map_collection',
    'IteratorFileStream',
    'get_file',
    'create_symlink_file',
    'ObjectStore',
    'ObjectStoreTransientError',
    'LibcloudObjectStore',
    'S3ObjectStore',
    'SFTPObjectStore',
    'MissingConditionalImportError',
    'import_object',
    'is_model_deepspeed',
    'is_model_fsdp',
    'is_notebook',
    'StringEnum',
    'load_checkpoint',
    'save_checkpoint',
    'ensure_folder_is_empty',
    'ensure_folder_has_no_conflicting_files',
    'export_for_inference',
    'export_with_logger',
    'quantize_dynamic',
    'format_name_with_dist',
    'format_name_with_dist_and_time',
    'is_tar',
    'batch_get',
    'batch_set',
    'configure_excepthook',
    'disable_env_report',
    'enable_env_report',
    'print_env',
    'retry',
    'model_eval_mode',
    'get_device',
    'is_tpu_installed',
]
