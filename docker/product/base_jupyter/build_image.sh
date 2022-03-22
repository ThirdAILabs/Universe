#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

# Build the slim image since we depend on it
REV_TAG=$(git log -1 --pretty=format:%h)
$BASEDIR/../slim/build_image.sh

cd $BASEDIR

docker build --build-arg REV_TAG=${REV_TAG} . -t thirdai_jupyter_release:${REV_TAG}
echo "================================================================="
echo "Built docker image thirdai_jupyter_release:${REV_TAG}"
echo "================================================================="