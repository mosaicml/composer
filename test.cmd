python -m composer.cli.launcher --stdout rank_{rank}_out -n 2 -m pytest -m gpu tests/trainer/test_fsdp.py -k test_fsdp_auto_microbatch
