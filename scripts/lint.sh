#!/usr/bin/env bash
set -exuo pipefail

# CD to the root directory of the repo
cd $(dirname $0)/..

# Install dependencies
python -m pip install .[all]

# Run isort (installed via setup.py)
python -m isort . -c

# Run yapf (installed via setup.py)
python -m yapf -dr . # not using -p since that can lead to race conditions

# Run docformatter
python -m docformatter -r --wrap-summaries 120 --wrap-descriptions 120 composer tests examples

# Install and run addlicense
# TODO(ravi): Switch to https://pypi.org/project/licenseheaders/ since it can be installed via setup.py and pip

curl -fsSL https://github.com/google/addlicense/releases/download/v1.0.0/addlicense_1.0.0_Linux_x86_64.tar.gz | \
    tar -xz -C /tmp

find . -type f -not -path '*/\.*' \( -iname \*.py -o -iname \*.pyi \) -print0 | \
    xargs -0 -n1 /tmp/addlicense -check -f ./LICENSE_HEADER

# Install and run pyright (requires nodejs to be already installed)

npm install pyright@1.1.217

./node_modules/.bin/pyright .
