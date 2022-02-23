# Tests

Composer uses PyTest as the testing framework.
The [Makefile](../Makefile) contains helper commands for invoking PyTest, as described below.

## Single-Rank Tests
To execute single rank tests, simply call `pytest` or `make test`.

## Multi-Rank Tests
By default, tests assume a world-size of one. However, tests which require multiple workers are annotated
with `@pytest.mark.world_size(n)`, where n is the number of workers required. Currently, only world sizes
of `1` and `2` are supported.

To run multi-rank tests, set the `WORLD_SIZE` flag, and use the `make test-dist` or `make test-dist-gpu`
targets (`make test` and `make test-gpu` ignore the `WORLD_SIZE` flag). 

When using `make test-dist` or `make test-dist-gpu`, the [composer launch script](../composer/cli/launcher.py) is used
to launch PyTest.

Valid options are `WORLD_SIZE=1` and `WORLD_SIZE=2`. For example:

* `make test-dist WORLD_SIZE=1 `: runs tests that have a world size of one using the composer launch script.
* `make test-dist WORLD_SIZE=2`: runs tests that have a world size of two
* `make test-dist-gpu WORLD_SIZE=2`: runs GPU tests that have a world size of two

## GPU Tests
GPU tests are marked with `@pytest.mark.gpu`. Tests without this annotation are assumed to be CPU-only tests.
By default, only CPU tests are run.

To run single-rank GPU tests, run `make test-gpu`.
To run multi-rank GPU tests, run `make test-dist-gpu WORLD_SIZE=2`.

## Test Duration
By default, all tests are run.
To run only "short" tests (those that take 2 seconds or less) or "long" tests (those that take more than 2 seconds),
set the `DURATION` flag to `short` or `long`, respectively.

The `DURATION` flag can be combined with other flags. for example:

* `make test DURATION=long ` runs long, single-rank tests.
* `make test-dist DURATION=all WORLD_SIZE=2 ` runs all tests that have a world size of two.
* `make test-gpu DURATION=all ` runs all single-rank GPU tests.

### Extra Arguments
To provide extra arguments to PyTest when using Makefile targets, set the `EXTRA_ARGS` flag.
For example `make test EXTRA_ARGS='-v'` runs PyTest with extra verbosity.
