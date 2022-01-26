#!/bin/bash

BASEDIR=$(dirname "$0")

# Make sure the code is built
"$BASEDIR/build.sh"

# Run tests with the passed in arguments
cd $BASEDIR/../build/
ctest "$@"
cd -