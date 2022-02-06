# several pytest settings
DURATION ?= all  # pytest duration, one of short, long or all
WORLD_SIZE ?= 1  # world size for DDP tests
EXTRA_ARGS ?=  # additional arguments

EXTRA_ARGS := --duration $(DURATION) $(EXTRA_ARGS)

dirs := composer examples tests

# run this to autoformat your code
style:
	isort -i $(dirs)
	yapf -ri $(dirs)
	docformatter -ri --wrap-summaries 120 --wrap-descriptions 120 $(dirs)

# this only checks for style & pyright, makes no code changes
lint:
	isort $(dirs) -c
	yapf -dr $(dirs)
	docformatter -r --wrap-summaries 120 --wrap-descriptions 120 $(dirs)
	pyright $(dirs)

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
