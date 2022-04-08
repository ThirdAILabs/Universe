#!/bin/bash

set -e

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

PLATFORM=$1

REV_TAG=$(git log -1 --pretty=format:%h)

cd $BASEDIR

echo "================================================================="
echo "Building docker thirdai_text_classification_interactive:${REV_TAG} with platform $PLATFORM"
echo "================================================================="
docker build --platform=$PLATFORM --build-arg REV_TAG=${REV_TAG} . -t thirdai_text_classification_interactive:${REV_TAG}  -f Interactive.Dockerfile
echo "================================================================="
echo "Built docker thirdai_text_classification_interactive:${REV_TAG} with platform $PLATFORM"
echo "================================================================="