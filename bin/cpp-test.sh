#!/bin/bash

BASEDIR=$(dirname "$0")

# Run tests with the passed in arguments
cd $BASEDIR/../build/
ctest "$@"