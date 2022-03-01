#!/usr/bin/env bash
set -euxo pipefail

# Tests for the `mosaicml/pytorch` docker image
# This script should be run from within the image
# It requires that composer has already been git
# cloned to the CWD from where the script is being run


##################################
# Verify pillow-simd is installed
# and that it does not get removed
# after a pip install pillow
##################################
pip install -y pillow
python -c "import PIL;assert 'post' in PIL.__version__"

COMPOSER_ROOT=$(abspath .)

###################################################
# Ensure that composer installs and can be imported
###################################################

test_composer () {
    # Go to a different folder so python won't attempt a relative import
    cd /tmp
    python -c "import composer"
    composer --help > /dev/null
    cd /COMPOSER_ROOT
}

# Normal install
pip install -y "$COMPOSER_ROOT"
test_composer()
pip uninstall -y composer

# Editable install
pip install -y -e "$COMPOSER_ROOT"
test_composer()
pip uninstall -y composer

# Normal install with dev
pip install -y "$COMPOSER_ROOT[dev]"
test_composer()
pip uninstall -y composer

# Inside virtualenv
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Normal install
pip install "$COMPOSER_ROOT"
test_composer()
pip uninstall -y composer

# Editable install
pip install -e "$COMPOSER_ROOT"
test_composer()
pip uninstall -y composer

# Remove the virtual environment
deactivate
rm -rf venv
