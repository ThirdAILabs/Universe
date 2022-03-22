#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

# Get the current branch to pass to git archive
# https://stackoverflow.com/questions/2111042/how-to-get-the-name-of-the-current-git-branch-into-a-variable-in-a-shell-script
CURRENT_BRANCH=$(git branch | sed -n -e 's/^\* \(.*\)/\1/p')
# Go out to Universe folder for archive
cd ../../../
# Archive the current branch into a zipped file. This can then be
# added directly into the first stage of DocSearch Docker build 
# (we don't include it in the final stage, instead we include
# just the .so file, see the DocSearch Dockerfile for more details).
git archive $CURRENT_BRANCH | bzip2 > $BASEDIR/Universe.tar.bz2

REV_TAG=$(git log -1 --pretty=format:%h)

cd $BASEDIR
docker build . -t thirdai_slim_release:$REV_TAG

echo "================================================================="
echo "Built docker image thirdai_slim_release:${REV_TAG}"
echo "================================================================="
