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

# TODO(josh): We need to build the license.serialized file here, using a private
# key stored locally (probably we will run this on our AWS machine and store 
# key there). For now license.serialized is just stored in our github repo, I've
# set it to expire in 90 days so in 90 days the smoke test will start failing, 
# this is not a permanent solution! Also dynamically generating it will allow
# us to write integration tests (right now we have no ASSERT_FAIL tests for
# the Dockerfile).

REV_TAG=$(git log -1 --pretty=format:%h)

cd $BASEDIR
echo "================================================================="
echo "Building docker image thirdai_slim:${REV_TAG} with platform $PLATFORM"
echo "================================================================="
docker build --platform=$PLATFORM . -t thirdai_slim:$REV_TAG
echo "================================================================="
echo "Built docker image thirdai_slim:${REV_TAG} with platform $PLATFORM"
echo "================================================================="
