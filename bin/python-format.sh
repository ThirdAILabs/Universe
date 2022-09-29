#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
black . --exclude "build|.venv|venv|.env|env|deps"
isort . --skip-glob deps --skip _deps