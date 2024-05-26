# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from composer import (
    algorithms,
    callbacks,
    core,
    devices,
    functional,
    loggers,
    loss,
    metrics,
    models,
    optim,
    profiler,
    trainer,
    utils,
)


# This very simple test is just to use the above imports, which check and make sure we can import all the top-level
# modules from composer. This is mainly useful for checking that we have correctly conditionally imported all optional
# dependencies.
def test_smoketest():
    assert callbacks
    assert algorithms
    assert core
    assert devices
    assert functional
    assert loggers
    assert loss
    assert metrics
    assert models
    assert optim
    assert profiler
    assert trainer
    assert utils
