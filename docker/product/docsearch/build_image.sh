#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

! docker inspect base_product > /dev/null 2>&1
BASE_IMAGE_DNE=$?
if [ $BASE_IMAGE_DNE ]
then
  $BASEDIR/../base/build_image.sh
fi

cd $BASEDIR
docker build . -t docsearch