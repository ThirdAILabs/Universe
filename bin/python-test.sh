#!/bin/bash

BASEDIR=$(dirname "$0")

# Run python tests. Switch to build directory first so we can use the so file.
# Forward all arguments.
cd $BASEDIR/../build/

# we look for tests in */python_tests/* to speed up test collection significantly
echo "Running the following tests:"
python3 -m pytest ../**/python_tests/ --ignore-glob=../deps --disable-warnings --collect-only "$@" 
echo "Output of the tests:"
python3 -m pytest ../**/python_tests/ --ignore-glob=../deps --disable-warnings -s "$@"