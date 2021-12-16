# Tests

Composer uses PyTest as the testing framework. 

## Testrunner
To run tests, we recommend that you use
[scripts/test.sh](../scripts/test.sh), which will automatically handle configuring the
world size correctly for multi-process tests. This test script passes through any CLI arguments
directly into `pytest`.

If you invoke pytest directly and not through `test.sh`, then only the 1-process tests will be executed.

## Custom pytest flags

### GPU Tests
By default, only CPU tests are run. GPU tests are marked with `@pytest.mark.gpu`.
To run GPU tests, run `./scripts/test.sh -m gpu`.


### Test duration
By default, only the "short" tests (i.e. those that take less than 2 seconds each) are run.
To run "long" tests, run `./scripts/test.sh --test_duration long`.
Or, to run all tests regardless of duration, run `./scripts/test.sh --test_duration all`
