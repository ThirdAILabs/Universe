#!/bin/bash

BASEDIR=$(dirname "$0")

# Run python tests. Switch to build directory first so we can use the so file.
# Forward all arguments.
cd $BASEDIR/../build/
python3 -m pytest ../ --ignore-glob=../deps -s "$@"