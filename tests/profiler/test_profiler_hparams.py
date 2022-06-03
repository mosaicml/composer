# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib

from composer.profiler import Profiler
from tests.common.hparams import assert_yaml_loads


def test_profiler_is_constructable_from_hparams(tmp_path: pathlib.Path):
    yaml_dict = {
        'schedule': {
            'cyclic': {
                'skip_first': 10
            }
        },
        'trace_handlers': {
            'json': {
                'folder': str(tmp_path / "traces")
            }
        },
        'torch_prof_profile_memory': False,
        'torch_prof_with_flops': False,
    }
    assert_yaml_loads(Profiler, yaml_dict, expected=Profiler)
