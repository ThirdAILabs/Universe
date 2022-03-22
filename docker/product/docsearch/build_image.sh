#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

# Only build the base image if it does not exist
REV_TAG=$(git log -1 --pretty=format:%h)
! docker inspect base_product:$REV_TAG > /dev/null 2>&1
BASE_IMAGE_DNE=$?
if [ $BASE_IMAGE_DNE ]
then
  $BASEDIR/../base/build_image.sh
fi

cd $BASEDIR
docker build --build-arg REV_TAG=${REV_TAG} . -t docsearch:${REV_TAG}