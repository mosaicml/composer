#!/bin/bash
set -exuo pipefail

composer -n 1 --master_port 26000 -m pytest $@
composer -n 2 --master_port 26000 -m pytest $@
composer -n 4 --master_port 26000 -m pytest $@
composer -n 8 --master_port 26000 -m pytest $@
