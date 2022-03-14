#!/bin/bash

BASEDIR=$(dirname "$0")

docker build -t thirdai/universe_dev_build -f $BASEDIR/../docker/develop/Dockerfile .
