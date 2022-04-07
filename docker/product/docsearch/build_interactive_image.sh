#!/bin/bash

set -e

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

PLATFORM=$1

# Build the base jupyter image since we depend on it
REV_TAG=$(git log -1 --pretty=format:%h)
$BASEDIR/../base_jupyter/build_image.sh $PLATFORM

# Download the files the container depends on 
# For now this will only work on blade
./download.sh

cd $BASEDIR

echo "================================================================="
echo "Building docker thirdai_docsearch_release:${REV_TAG} with platform $PLATFORM"
echo "================================================================="
docker build --platform=$PLATFORM --build-arg REV_TAG=${REV_TAG} . -t thirdai_docsearch_release:${REV_TAG} -f Interactive.Dockerfile
echo "================================================================="
echo "Built docker thirdai_docsearch_release:${REV_TAG} with platform $PLATFORM"
echo "================================================================="
