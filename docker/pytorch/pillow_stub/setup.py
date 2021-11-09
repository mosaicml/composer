import os

from setuptools import setup

setup(
    name="pillow",
    version=os.environ['PILLOW_PSEUDOVERSION'],
)
