#!/bin/bash

BASEDIR=$(dirname "$0")

# Make sure the code is built
"$BASEDIR/build.sh"

# Run all tests
ctest --test-dir "$BASEDIR/../build"


