#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

base/build_image.sh
docsearch/build_image.sh