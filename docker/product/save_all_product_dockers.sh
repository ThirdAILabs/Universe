#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

./build_all_product_dockers.sh

REV_TAG=$(git log -1 --pretty=format:%h)
docker save -o $BASEDIR/thirdai_docsearch_release.tar thirdai_docsearch_release:$REV_TAG
gzip $BASEDIR/thirdai_docsearch_release.tar
