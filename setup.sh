#!/bin/bash
# Build the Cython/C++ extensions and install the package in editable mode.
#
# Run this inside the conda environment created from protein.yml.
#
# `pip install -e .` rather than `python setup.py develop`: the latter resolves
# dependencies through easy_install, which prefers source tarballs and fails on
# modern ones. --no-build-isolation is required because setup.py imports torch
# to build the CUDA extensions, and an isolated build environment would not
# have it.
set -euo pipefail

pip install Cython
python setup.py build_ext --inplace
pip install -e . --no-build-isolation
