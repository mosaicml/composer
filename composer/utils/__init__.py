# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helper utilities."""

from composer.utils.auto_log_hparams import (convert_flat_dict_to_nested_dict, convert_nested_dict_to_flat_dict,
                                             extract_hparams)
from composer.utils.batch_helpers import batch_get, batch_set
from composer.utils.checkpoint import PartialFilePath, load_checkpoint, safe_torch_load, save_checkpoint
from composer.utils.collect_env import (configure_excepthook, disable_env_report, enable_env_report,
                                        get_composer_env_dict, print_env)
from composer.utils.device import get_device, is_hpu_installed, is_tpu_installed
from composer.utils.eval_client import EvalClient, LambdaEvalClient, LocalEvalClient, MosaicMLLambdaEvalClient
from composer.utils.file_helpers import (FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, FORMAT_NAME_WITH_DIST_TABLE,
                                         create_symlink_file, ensure_folder_has_no_conflicting_files,
                                         ensure_folder_is_empty, format_name_with_dist, format_name_with_dist_and_time,
                                         get_file, is_tar, maybe_create_object_store_from_uri,
                                         maybe_create_remote_uploader_downloader_from_uri, parse_uri)
from composer.utils.import_helpers import MissingConditionalImportError, import_object
from composer.utils.inference import ExportFormat, Transform, export_for_inference, export_with_logger, quantize_dynamic
from composer.utils.iter_helpers import IteratorFileStream, ensure_tuple, map_collection
from composer.utils.misc import (create_interval_scheduler, get_free_tcp_port, is_model_deepspeed, is_model_fsdp,
                                 is_notebook, model_eval_mode, using_torch_2)
from composer.utils.object_store import (GCSObjectStore, LibcloudObjectStore, ObjectStore, ObjectStoreTransientError,
                                         OCIObjectStore, S3ObjectStore, SFTPObjectStore, UCObjectStore)
from composer.utils.retrying import retry
from composer.utils.string_enum import StringEnum

__all__ = [
    'ensure_tuple',
    'get_free_tcp_port',
    'map_collection',
    'IteratorFileStream',
    'FORMAT_NAME_WITH_DIST_AND_TIME_TABLE',
    'FORMAT_NAME_WITH_DIST_TABLE',
    'get_file',
    'PartialFilePath',
    'create_symlink_file',
    'ObjectStore',
    'ObjectStoreTransientError',
    'LibcloudObjectStore',
    'S3ObjectStore',
    'SFTPObjectStore',
    'OCIObjectStore',
    'GCSObjectStore',
    'UCObjectStore',
    'MissingConditionalImportError',
    'import_object',
    'is_model_deepspeed',
    'is_model_fsdp',
    'is_notebook',
    'StringEnum',
    'load_checkpoint',
    'save_checkpoint',
    'safe_torch_load',
    'ensure_folder_is_empty',
    'ensure_folder_has_no_conflicting_files',
    'export_for_inference',
    'export_with_logger',
    'quantize_dynamic',
    'format_name_with_dist',
    'format_name_with_dist_and_time',
    'is_tar',
    'maybe_create_object_store_from_uri',
    'maybe_create_remote_uploader_downloader_from_uri',
    'parse_uri',
    'batch_get',
    'batch_set',
    'configure_excepthook',
    'disable_env_report',
    'enable_env_report',
    'print_env',
    'get_composer_env_dict',
    'retry',
    'model_eval_mode',
    'get_device',
    'is_tpu_installed',
    'is_hpu_installed',
    'ExportFormat',
    'Transform',
    'export_with_logger',
    'extract_hparams',
    'convert_nested_dict_to_flat_dict',
    'convert_flat_dict_to_nested_dict',
    'using_torch_2',
    'create_interval_scheduler',
    'EvalClient',
    'LambdaEvalClient',
    'LocalEvalClient',
    'MosaicMLLambdaEvalClient',
]
