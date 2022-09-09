#!/bin/bash

BASEDIR=$(dirname "$0")

cd $BASEDIR/../

docker build -t thirdai/universe_dev_build -f docker/benchmarking/Dockerfile .
