import pathlib

from composer.profiler import Profiler
from tests.common import assert_is_constructable_from_yaml


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
    assert_is_constructable_from_yaml(Profiler, yaml_dict, expected=Profiler)
