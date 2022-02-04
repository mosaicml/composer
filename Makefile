style:
	python -m isort .
	python -m yapf -dr .
	python -m docformatter -rc --wrap-summaries 120 --wrap-descriptions 120 composer tests examples

license:
	# TODO (ravi): Switch to https://pypi.org/project/licenseheaders/ since it can be installed via setup.py and pip
	curl -fsSL https://github.com/google/addlicense/releases/download/v1.0.0/addlicense_1.0.0_Linux_x86_64.tar.gz | \
    	tar -xz -C /tmp
	find . -type f -not -path '*/\.*' \( -iname \*.py -o -iname \*.pyi \) -print0 | \
	    xargs -0 -n1 /tmp/addlicense -check -f ./LICENSE_HEADER

quality:
	# requires pyright to be installed
	pyright .

# run only short CPU tests
test:
	pytest tests/

# run all tests, including mgpu tests
# uses the composer launcher script to properly configure
# mgpu cases
test-mgpu:
	python -m composer.cli.launcher -n 1 -m pytest --test_duration all
	python -m composer.cli.launcher -n 2 -m pytest --test_duration all

