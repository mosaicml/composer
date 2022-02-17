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

* `WORLD_SIZE=1 make test-dist`: runs tests that have a world size of one using the composer launch script.
* `WORLD_SIZE=2 make test-dist`: runs tests that have a world size of two
* `WORLD_SIZE=2 make test-dist-gpu`: runs GPU tests that have a world size of two

## GPU Tests
GPU tests are marked with `@pytest.mark.gpu`. Tests without this annotation are assumed to be CPU-only tests.
By default, only CPU tests are run.

To run single-rank GPU tests, run `make test-gpu`.
To run multi-rank GPU tests, run `WORLD_SIZE=2 make test-dist-gpu`.

## Test Duration
By default, only the "short" tests (i.e. those that take less than 2 seconds each) are run.
To run "long" tests, set the `TEST_DURATION` flag before calling `make`.
Valid options are `short` (the default) for tests that take 2 seconds or less; `long` for tests that take
longer than 2 seconds, or `all` for all tests.

The `TEST_DURATION` flag can be combined with other flags. for example:

* `TEST_DURATION=long make test` runs long, single-rank tests
* `TEST_DURATION=all WORLD_SIZE=2 make test-dist` runs all tests that have a world size of two
* `TEST_DURATION=all make test-gpu` runs all single-rank GPU tests.

### Extra Arguments
To provide extra arguments to PyTest when using Makefile targets, set the `EXTRA_ARGS` flag.
For example `EXTRA_ARGS='-v' make test` runs PyTest with extra verbosity.
