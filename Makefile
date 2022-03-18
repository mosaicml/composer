# several pytest settings
DURATION ?= all  # pytest duration, one of short, long or all
WORLD_SIZE ?= 1  # world size for launcher tests
MASTER_PORT ?= 26000 # port for distributed tests
PYTHON ?= python  # Python command
PYTEST ?= pytest  # Pytest command
PYRIGHT ?= pyright  # Pyright command. Pyright must be installed seperately -- e.g. `node install -g pyright`
EXTRA_ARGS ?=  # extra arguments for pytest

# Force append the duration flag to extra args
override EXTRA_ARGS += --duration $(DURATION)

dirs := composer examples tests

# run this to autoformat your code
style:
	$(PYTHON) -m isort $(dirs)
	$(PYTHON) -m yapf -rip $(dirs)
	$(PYTHON) -m docformatter -ri --wrap-summaries 120 --wrap-descriptions 120 $(dirs)

# this only checks for style & pyright, makes no code changes
lint:
	$(PYTHON) -m isort -c --diff $(dirs)
	$(PYTHON) -m yapf -dr $(dirs)
	$(PYTHON) -m docformatter -r --wrap-summaries 120 --wrap-descriptions 120 $(dirs)
	$(PYRIGHT) $(dirs)

test:
	$(PYTHON) -m $(PYTEST) $(EXTRA_ARGS)

test-gpu:
	$(PYTHON) -m $(PYTEST) -m gpu $(EXTRA_ARGS)

# runs tests with the launcher
test-dist:
	$(PYTHON) -m composer.cli.launcher -n $(WORLD_SIZE) --master_port $(MASTER_PORT) -m $(PYTEST) $(EXTRA_ARGS)

test-dist-gpu:
	$(PYTHON) -m composer.cli.launcher -n $(WORLD_SIZE) --master_port $(MASTER_PORT) -m $(PYTEST) -m gpu $(EXTRA_ARGS)

clean-notebooks:
	$(PYTHON) scripts/clean_notebooks.py -i notebooks/*.ipynb

.PHONY: test test-gpu test-dist test-dist-gpu lint style clean-notebooks
