# Contributing to Composer

Thanks for considering contributing to Composer!

Issues tagged with [good first issue](https://github.com/mosaicml/composer/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) are great options to start contributing.

If you have questions, join us on [Slack](https://join.slack.com/t/mosaicml-community/shared_invite/zt-w0tiddn9-WGTlRpfjcO9J5jyrMub1dg) -- we'll be happy to help you!

We welcome contributions for bug fixes, new efficient methods you'd like to contribute to the community, or new models and datasets!

## New Algorithms

Have a new algorithm you'd like to contribute to the library as part of your research? We welcome any PRs, and recommend filing an issue with the proposed method or reaching out on Slack first!

## Prerequisites

To set up the development environment in your local box, run the command below. 

1\. This should bring most the dependencies needed for testing and linting the code:

```bash
pip install -e .[dev]
```

2\. Next, you need to install `pyright` (we have not included it in `setup.py` because we have found that pyright is unreliable when it comes to version pinning). For this, you will need to first install NodeJS, more instructions can be found at https://nodejs.org/en/download/package-manager/. Then, to install pyright run:

```bash
# NOTE. You might need sudo permissions.
npm install -g pyright@1.1.224
```

## Submitting a contribution

To submit a contribution:

1\. Fork a copy of the [Composer](https://github.com/mosaicml/composer) library to your own account.

2\. Clone your fork locally and add the mosaicml repo as a remote repository:

```bash
git clone git@github.com:<github_id>/composer.git
cd composer
git remote add upstream https://github.com/mosaicml/composer.git
```

3\. Create a branch and make your proposed changes.

```bash
git checkout -b cool-new-feature
```

4\. We use a few formatters for code style, and you run the following to autocorrect your files:

```bash
make style
```

That will run the [yapf](https://github.com/google/yapf) formatter for general formatting,
[isort](https://github.com/PyCQA/isort) to sort imports, and
[docformatter](https://github.com/myint/docformatter) to format docstrings.

5\. When you are ready, submit a pull request into the composer repository! If merged, we'll reach out to send you some free swag :)

## Running Tests

To test your changes locally, run:

1. `make test`  # run CPU tests
1. `make test-gpu`  # run GPU tests
1. `cd docs && make doctest`  # run doctests

Some of our checks test distributed training as well. To test these, run:

* `make test-dist WORLD_SIZE=2`  # run 2-cpu distributed tests
* `make test-dist-gpu WORLD_SIZE=2`  # run 2-gpu distributed tests

These tests run with the `composer` launcher. We also support `WORLD_SIZE=1`, which would run the tests with the `composer` launcher on a single device.

See the [Makefile](https://github.com/mosaicml/composer/blob/dev/Makefile) for more information.

## Code Style & Typing

Follow Google's
[Python Style Guide](https://google.github.io/styleguide/pyguide.html) for how to format and structure code. Many of these guidelines are already taken care of by the `make style` command above.

Composer aims to annotate all functions with type annotations (introduced in
[PEP 526](https://www.python.org/dev/peps/pep-0526/)). Don't worry if you are not a Python typing expert; put in the pull request, and we'll help you with getting the code into shape.
