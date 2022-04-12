#!/bin/bash

set -e

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

PLATFORM=$1

REV_TAG=$(git log -1 --pretty=format:%h)

# Download the files the container depends on 
# For now this will only work on blade
./download.sh

cd $BASEDIR

echo "================================================================="
echo "Building docker thirdai_docsearch_inference:${REV_TAG} with platform $PLATFORM"
echo "================================================================="
docker build --platform=$PLATFORM --build-arg REV_TAG=${REV_TAG} . -t thirdai_docsearch_inference:${REV_TAG} -f Inference.Dockerfile
echo "================================================================="
echo "Built docker thirdai_docsearch_inference:${REV_TAG} with platform $PLATFORM"
echo "================================================================="
