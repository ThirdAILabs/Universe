#!/bin/bash

set -e

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

PLATFORM=$1

# Go out to Universe folder for archive
cd ../../../

# Archive the current branch into a zipped file. This can then be
# added directly into the first stage of DocSearch Docker build 
# (we don't include it in the final stage, instead we include
# just the .so file, see the DocSearch Dockerfile for more details).
git-archive-all $BASEDIR/Universe.tar.bz2

REV_TAG=$(git log -1 --pretty=format:%h)

cd $BASEDIR
echo "================================================================="
echo "Building docker image thirdai_slim_release:${REV_TAG} with platform $PLATFORM"
echo "================================================================="
docker build --platform=$PLATFORM . -t thirdai_slim_release:$REV_TAG
echo "================================================================="
echo "Built docker image thirdai_slim_release:${REV_TAG} with platform $PLATFORM"
echo "================================================================="
