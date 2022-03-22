#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

slim/build_image.sh
base_jupyter/build_image.sh
docsearch/build_image.sh