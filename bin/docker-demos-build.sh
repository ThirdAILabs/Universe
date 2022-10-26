#!/bin/bash

BASEDIR=$(dirname "$0")

cd $BASEDIR/../

docker build -t thirdai/run_demos_build -f docker/demos/Dockerfile .
