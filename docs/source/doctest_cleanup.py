import os

# variables are defined in doctest_fixtures.py
# pyright: reportUndefinedVariable=none

# tmpdir and cwd were defined in doctest_fixtures.py

os.chdir(cwd)

tmpdir.cleanup()
