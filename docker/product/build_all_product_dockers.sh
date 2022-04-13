#!/bin/bash

set -e

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

slim/build_image.sh $1
base_jupyter/build_interactive_image.sh $1
docsearch/build_interactive_image.sh $1
docsearch/build_inference_image.sh $1
text_classification/build_interactive_image.sh $1

# Ensure all images built correctly
REV_TAG=$(git log -1 --pretty=format:%h)
if [ $(docker images | grep $REV_TAG | wc -l) != "5" ]
then
  echo "ERROR: Not all images were built"
  exit 1
fi