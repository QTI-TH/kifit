#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="qiss",
    version="4.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
)
