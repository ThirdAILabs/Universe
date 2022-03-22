#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

# Build the base image since we depend on it (this may do nothing)
REV_TAG=$(git log -1 --pretty=format:%h)
$BASEDIR/../base/build_image.sh

cd $BASEDIR

docker build --build-arg REV_TAG=${REV_TAG} . -t docsearch:${REV_TAG}
echo "================================================================="
echo "Built docker image docsearch:${REV_TAG}"
echo "================================================================="