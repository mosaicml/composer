# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helper utilities."""

from composer.utils.auto_log_hparams import (
    convert_flat_dict_to_nested_dict,
    convert_nested_dict_to_flat_dict,
    extract_hparams,
)
from composer.utils.batch_helpers import batch_get, batch_set
from composer.utils.checkpoint import (
    PartialFilePath,
    get_save_filename,
    load_checkpoint,
    safe_torch_load,
    save_checkpoint,
)
from composer.utils.collect_env import (
    configure_excepthook,
    disable_env_report,
    enable_env_report,
    get_composer_env_dict,
    print_env,
)
from composer.utils.compression import (
    KNOWN_COMPRESSORS,
    CliCompressor,
    get_compressor,
    is_compressed_pt,
)
from composer.utils.device import get_device, is_hpu_installed, is_xla_installed
from composer.utils.eval_client import EvalClient, LambdaEvalClient, LocalEvalClient
from composer.utils.file_helpers import (
    FORMAT_NAME_WITH_DIST_AND_TIME_TABLE,
    FORMAT_NAME_WITH_DIST_TABLE,
    create_symlink_file,
    ensure_folder_has_no_conflicting_files,
    ensure_folder_is_empty,
    extract_path_from_symlink,
    format_name_with_dist,
    format_name_with_dist_and_time,
    get_file,
    is_tar,
    maybe_create_object_store_from_uri,
    maybe_create_remote_uploader_downloader_from_uri,
    parse_uri,
    validate_credentials,
)
from composer.utils.import_helpers import MissingConditionalImportError, import_object
from composer.utils.inference import ExportFormat, Transform, export_for_inference, export_with_logger, quantize_dynamic
from composer.utils.iter_helpers import IteratorFileStream, ensure_tuple, map_collection
from composer.utils.misc import (
    STR_TO_DTYPE,
    ParallelismType,
    add_vision_dataset_transform,
    create_interval_scheduler,
    get_free_tcp_port,
    is_model_fsdp,
    is_notebook,
    model_eval_mode,
    partial_format,
)
from composer.utils.object_store import (
    MLFLOW_EXPERIMENT_ID_FORMAT_KEY,
    MLFLOW_RUN_ID_FORMAT_KEY,
    GCSObjectStore,
    LibcloudObjectStore,
    MLFlowObjectStore,
    ObjectStore,
    ObjectStoreTransientError,
    OCIObjectStore,
    S3ObjectStore,
    SFTPObjectStore,
    UCObjectStore,
    build_remote_backend,
)
from composer.utils.parallelism import FSDPConfig, ParallelismConfig, TPConfig
from composer.utils.remote_uploader import RemoteFilesExistingCheckStatus, RemoteUploader
from composer.utils.retrying import retry
from composer.utils.string_enum import StringEnum
from composer.utils.warnings import VersionedDeprecationWarning

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
    'MLFlowObjectStore',
    'OCIObjectStore',
    'GCSObjectStore',
    'UCObjectStore',
    'MissingConditionalImportError',
    'get_save_filename',
    'import_object',
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
    'is_xla_installed',
    'is_hpu_installed',
    'ExportFormat',
    'Transform',
    'export_with_logger',
    'extract_hparams',
    'convert_nested_dict_to_flat_dict',
    'convert_flat_dict_to_nested_dict',
    'create_interval_scheduler',
    'EvalClient',
    'LambdaEvalClient',
    'LocalEvalClient',
    'partial_format',
    'add_vision_dataset_transform',
    'VersionedDeprecationWarning',
    'is_compressed_pt',
    'CliCompressor',
    'get_compressor',
    'KNOWN_COMPRESSORS',
    'STR_TO_DTYPE',
    'ParallelismType',
    'FSDPConfig',
    'TPConfig',
    'ParallelismConfig',
    'MLFLOW_EXPERIMENT_ID_FORMAT_KEY',
    'MLFLOW_RUN_ID_FORMAT_KEY',
    'extract_path_from_symlink',
    'RemoteUploader',
    'validate_credentials',
    'build_remote_backend',
    'RemoteFilesExistingCheckStatus',
]
