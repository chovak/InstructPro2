#!/bin/bash
# Build the Cython extensions and install the package in development mode.
set -euo pipefail

pip3 install Cython
python3 setup.py build_ext --inplace
python3 setup.py develop --user
