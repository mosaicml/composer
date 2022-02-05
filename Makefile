# several pytest settings
DURATION ?= all  # pytest duration, one of short, long or all
WORLD_SIZE ?= 1  # world size for DDP tests
EXTRA_ARGS ?=  # additional arguments

EXTRA_ARGS := --duration $(DURATION) $(EXTRA_ARGS)

style:
	python -m isort . -cv
	python -m yapf -dr .
	python -m docformatter -rc --wrap-summaries 120 --wrap-descriptions 120 composer tests examples

license:
	# TODO (ravi): Switch to https://pypi.org/project/licenseheaders/ since it can be installed via setup.py and pip
	curl -fsSL https://github.com/google/addlicense/releases/download/v1.0.0/addlicense_1.0.0_Linux_x86_64.tar.gz | \
    	tar -xz -C /tmp
	find . -type f -not -path '*/\.*' \( -iname \*.py -o -iname \*.pyi \) -print0 | \
	    xargs -0 -n1 /tmp/addlicense -check -f ./LICENSE_HEADER

typing:
	pyright .

test:
	pytest tests/ $(EXTRA_ARGS)

test-gpu:
	pytest tests/ -m gpu $(EXTRA_ARGS)

test-deepspeed:
	pytest tests/ -m deepspeed $(EXTRA_ARGS)

# run all tests, including multi-gpu tests
# uses the composer launcher script
test-with-ddp:
	python -m composer.cli.launcher -n ${WORLD_SIZE} -m pytest $(EXTRA_ARGS)

test-all: test test-gpu test-deepspeed test-with-ddp

.PHONY: test test-gpu test-with-ddp test-deepspeed
