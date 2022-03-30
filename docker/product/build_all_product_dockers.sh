#!/bin/bash

set -e

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

slim/build_image.sh $1
base_jupyter/build_image.sh $1
docsearch/build_image.sh $1
text_classification/build_image.sh $1