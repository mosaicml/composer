# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from composer.trainer.trainer import _generate_run_name
from composer.utils import reproducibility


def test_different_run_names():
    """Verify runs with the same seed generate different names"""
    reproducibility.seed_all(42)
    name1 = _generate_run_name()
    reproducibility.seed_all(42)
    name2 = _generate_run_name()
    assert name1 != name2
