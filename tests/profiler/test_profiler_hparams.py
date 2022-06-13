# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib

from composer.profiler import Profiler
from tests.common.hparams import construct_from_yaml


def test_profiler_is_constructable_from_hparams(tmp_path: pathlib.Path):
    yaml_dict = {
        'schedule': {
            'cyclic': {
                'skip_first': 10
            }
        },
        'trace_handlers': {
            'json': {
                'folder': str(tmp_path / 'traces')
            }
        },
        'torch_prof_profile_memory': False,
        'torch_prof_with_flops': False,
    }
    assert isinstance(construct_from_yaml(Profiler, yaml_dict), Profiler)
