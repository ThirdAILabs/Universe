#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

./build_all_product_dockers.sh

docker save -o $BASEDIR/docsearch.tar docsearch
gzip $BASEDIR/docsearch.tar
