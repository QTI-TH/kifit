#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="kifit",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
)
