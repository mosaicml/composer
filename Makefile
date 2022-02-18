# several pytest settings
DURATION ?= short  # pytest duration, one of short, long or all
WORLD_SIZE ?= 1  # world size for launcher tests
EXTRA_ARGS ?=  # additional arguments
PYTHON ?= python
PYTEST ?= pytest
MASTER_PORT ?= 26000 # port for distributed tests


EXTRA_ARGS := --duration $(DURATION) $(EXTRA_ARGS)

dirs := composer examples tests

# run this to autoformat your code
style:
	$(PYTHON) -m isort -i $(dirs)
	$(PYTHON) -m yapf -rip $(dirs)
	$(PYTHON) -m docformatter -ri --wrap-summaries 120 --wrap-descriptions 120 $(dirs)

# this only checks for style & pyright, makes no code changes
lint:
	$(PYTHON) -m isort $(dirs) -c
	$(PYTHON) -m yapf -dr $(dirs)
	$(PYTHON) -m docformatter -r --wrap-summaries 120 --wrap-descriptions 120 $(dirs)
	pyright $(dirs)

test:
	$(PYTHON) -m $(PYTEST) tests/ $(EXTRA_ARGS)

test-gpu:
	$(PYTHON) -m $(PYTEST) tests/ -m gpu $(EXTRA_ARGS)

# runs tests with the launcher
test-dist:
	$(PYTHON) -m composer.cli.launcher -n $(WORLD_SIZE) --master_port $(MASTER_PORT) -m $(PYTEST) $(EXTRA_ARGS)

test-dist-gpu:
	$(PYTHON) -m composer.cli.launcher -n $(WORLD_SIZE) --master_port $(MASTER_PORT) -m $(PYTEST) -m gpu $(EXTRA_ARGS)

clean-notebooks:
	$(PYTHON) scripts/clean_notebooks.py -i notebooks/*.ipynb

.PHONY: test test-gpu test-dist test-dist-gpu lint style clean-notebooks
